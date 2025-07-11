# ROI Mask策略分析与改进方案

## 1. 当前ROI Mask策略概述

### 1.1 ROI Mask配置位置

**训练脚本**: `det_private_base_exp_8v1l_y300x32_deformable_120m_400q_refine_28w_fixoptim.py`
- 第178行: `self.model_cfg["det_head"]["use_roi_mask"] = True`

**数据配置**: `det_annos_hf_200m_32m_8v5r1l_mmL_chengqu_Z10_new_fovrange_120.py`
- 第200行: `roi_mask=[-32, -80, 32, 120]`  # [x_min, y_min, x_max, y_max]

### 1.2 ROI Mask的作用机制

ROI Mask定义了一个矩形区域，用于：
1. **限制训练范围**: 只有在ROI区域内的query才参与正负样本分配
2. **蒙版匹配策略**: 区域外的query被标记为"don't care"(-1)，不参与损失计算

## 2. 当前实现的问题分析

### 2.1 单目标蒙版处理

**当前蒙版处理逻辑** (`base.py` 203-233行):
```python
# 只处理单个目标的蒙版标记
if self.occlusion_threshold == 1 and not self._get_occlusion_attr(annotation):
    category = "蒙版"  # 单个目标标记为蒙版
```

**问题**: 
- 只能处理单个目标级别的蒙版
- 无法处理大片区域的蒙版（如施工区域、事故现场等）
- 缺乏空间连续性的蒙版处理

### 2.2 ROI Mask的局限性

**当前ROI Mask实现** (`hungarian_assigner_3d.py` 108-116行):
```python
if roi_mask is not None:
    is_in_roi = (
        (bbox_pred[:, 0] > roi_mask[0]) *  # x > x_min
        (bbox_pred[:, 0] < roi_mask[2]) *  # x < x_max  
        (bbox_pred[:, 1] > roi_mask[1]) *  # y > y_min
        (bbox_pred[:, 1] < roi_mask[4])    # y < y_max
    )
    in_roi_idx = is_in_roi
```

**问题**:
- 只支持单一矩形ROI区域
- 无法处理复杂形状的蒙版区域
- 无法处理多个分离的蒙版区域

## 3. 大片蒙版区域处理的改进方案

### 3.1 方案1: 多ROI Mask支持

**修改数据配置**:
```python
# 支持多个ROI区域
roi_mask = [
    [-32, -80, 32, 120],      # 主要检测区域
    [-50, 100, 50, 150],      # 额外检测区域
]

# 或者支持蒙版区域定义
mask_areas = [
    [-10, 50, 10, 80],        # 施工区域蒙版
    [20, 30, 40, 60],         # 事故区域蒙版
]
```

**修改HungarianAssigner3D**:
```python
def assign(self, bbox_pred, cls_pred, gt_bboxes, gt_labels, 
           roi_mask=None, mask_areas=None, query_bboxes=None, **kwargs):
    
    # 处理多ROI区域
    if isinstance(roi_mask, list) and len(roi_mask) > 0:
        if isinstance(roi_mask[0], list):  # 多个ROI区域
            in_roi_idx = torch.zeros(bbox_pred.shape[0], dtype=torch.bool, device=bbox_pred.device)
            for roi in roi_mask:
                roi_mask_single = (
                    (bbox_pred[:, 0] > roi[0]) * (bbox_pred[:, 0] < roi[2]) *
                    (bbox_pred[:, 1] > roi[1]) * (bbox_pred[:, 1] < roi[3])
                )
                in_roi_idx |= roi_mask_single
        else:  # 单个ROI区域
            in_roi_idx = (
                (bbox_pred[:, 0] > roi_mask[0]) * (bbox_pred[:, 0] < roi_mask[2]) *
                (bbox_pred[:, 1] > roi_mask[1]) * (bbox_pred[:, 1] < roi_mask[3])
            )
    
    # 处理蒙版区域
    if mask_areas is not None and len(mask_areas) > 0:
        in_mask_idx = torch.zeros(bbox_pred.shape[0], dtype=torch.bool, device=bbox_pred.device)
        for mask_area in mask_areas:
            mask_single = (
                (bbox_pred[:, 0] > mask_area[0]) * (bbox_pred[:, 0] < mask_area[2]) *
                (bbox_pred[:, 1] > mask_area[1]) * (bbox_pred[:, 1] < mask_area[3])
            )
            in_mask_idx |= mask_single
        
        # 蒙版区域内的query标记为don't care
        in_roi_idx = in_roi_idx & (~in_mask_idx)
```

### 3.2 方案2: 基于多边形的蒙版区域

**数据配置**:
```python
# 支持多边形蒙版区域
polygon_mask_areas = [
    {
        "type": "polygon",
        "points": [[-10, 50], [10, 50], [15, 80], [-15, 80]],  # 不规则施工区域
        "mask_type": "construction"
    },
    {
        "type": "circle", 
        "center": [30, 40],
        "radius": 20,  # 圆形事故区域
        "mask_type": "accident"
    }
]
```

**实现多边形蒙版检查**:
```python
def points_in_polygon_mask(points, polygon_masks):
    """检查点是否在多边形蒙版区域内"""
    in_mask = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    
    for mask_config in polygon_masks:
        if mask_config["type"] == "polygon":
            # 使用射线法判断点是否在多边形内
            mask_points = torch.tensor(mask_config["points"], device=points.device)
            in_polygon = points_in_poly(points, mask_points)
            in_mask |= torch.from_numpy(in_polygon).to(points.device)
            
        elif mask_config["type"] == "circle":
            center = torch.tensor(mask_config["center"], device=points.device)
            radius = mask_config["radius"]
            dist = torch.norm(points - center, dim=1)
            in_circle = dist < radius
            in_mask |= in_circle
    
    return in_mask
```

### 3.3 方案3: 基于语义分割的蒙版区域

**集成语义分割结果**:
```python
def get_semantic_mask_areas(semantic_map, mask_classes=["construction", "accident"]):
    """从语义分割结果中提取蒙版区域"""
    mask_areas = []
    
    for mask_class in mask_classes:
        class_id = SEMANTIC_CLASS_MAP[mask_class]
        class_mask = (semantic_map == class_id)
        
        # 提取连通区域
        contours = cv2.findContours(class_mask.astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 将像素坐标转换为世界坐标
            world_polygon = pixel_to_world_coords(contour, camera_params)
            mask_areas.append({
                "type": "polygon",
                "points": world_polygon,
                "mask_type": mask_class
            })
    
    return mask_areas
```

## 4. 具体修改步骤

### 4.1 修改数据配置

**文件**: `det_annos_hf_200m_32m_8v5r1l_mmL_chengqu_Z10_new_fovrange_120.py`

```python
# 原配置
roi_mask=[-32, -80, 32, 120],

# 新配置 - 支持多区域和蒙版
roi_mask=[-32, -80, 32, 120],  # 主检测区域
mask_areas=[  # 蒙版区域列表
    [-10, 50, 10, 80],   # 施工区域1
    [20, 30, 40, 60],    # 施工区域2
],
```

### 4.2 修改HungarianAssigner3D

**文件**: `perceptron/layers/head/det3d/target_assigner/hungarian_assigner_3d.py`

在`assign`方法中添加mask_areas参数和处理逻辑。

### 4.3 修改SparseE2EHead

**文件**: `perceptron/layers/head/det3d/sparse_head.py`

在目标分配调用中传递mask_areas参数：

```python
assign_results = self.assigner.assign(
    bbox_pred, logits_pred, gt_bboxes, gt_labels,
    roi_mask=roi_mask,
    mask_areas=mask_areas,  # 新增参数
    query_bboxes=query_bboxes,
    fov_boardline=fov_boardline,
)
```

### 4.4 修改数据加载

**文件**: `perceptron/data/det3d/private/private_multimodal.py`

在数据字典中添加mask_areas：

```python
data_dict["mask_areas"] = np.asarray(self.mask_areas) if hasattr(self, 'mask_areas') else None
```

## 5. 预期效果

### 5.1 解决的问题

1. **大片蒙版区域处理**: 支持施工区域、事故现场等大片区域的蒙版
2. **多区域蒙版**: 支持同时处理多个分离的蒙版区域
3. **复杂形状蒙版**: 支持多边形、圆形等复杂形状的蒙版区域
4. **动态蒙版**: 可以根据语义分割结果动态生成蒙版区域

### 5.2 训练改进

1. **更精确的负样本**: 蒙版区域内的query不会被错误地标记为负样本
2. **更好的收敛**: 避免在蒙版区域产生错误的监督信号
3. **更强的泛化**: 模型学会忽略不确定或不可靠的区域

## 6. 实施建议

1. **渐进式实施**: 先实现方案1（多矩形ROI），再扩展到方案2（多边形）
2. **向后兼容**: 保持对现有单ROI配置的兼容性
3. **充分测试**: 在小规模数据上验证改进效果
4. **性能监控**: 确保新的蒙版处理不会显著影响训练速度

这个改进方案可以有效解决当前蒙版匹配策略只考虑单个目标的局限性，为大片蒙版区域提供更好的处理能力。
