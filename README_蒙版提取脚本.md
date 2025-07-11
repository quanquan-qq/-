# 蒙版标注提取脚本使用说明

## 概述

本脚本用于从指定的四个数据集中提取包含蒙版标注的JSON文件：
- `z10_label_1230_train`
- `Z10_label_0207_7w`
- `z1_label_1230_bmk_qy`
- `BMK_02`

## 蒙版检测逻辑

根据 `base.py` 中的逻辑，蒙版类别通过以下条件生成：

1. **直接蒙版类别**：标注的 `category` 字段直接包含蒙版相关类别
   - `"蒙版"`, `"mask"`, `"masked_area"`, `"正向蒙版"`, `"负向蒙版"`

2. **遮挡阈值检查失败**：当 `occlusion_threshold > 0` 且对象在相机中不可见时

3. **异常值过滤**：当 `filter_outlier_boxes = True` 且对象尺寸超出合理范围时
   - 行人类别：长宽高任一维度 > 3米
   - 车辆类别：长宽高超出 [30, 6, 10] 米

4. **短轨迹过滤**：当 `filter_short_track = True` 且轨迹长度 < 2帧时

## 文件说明

### 主要脚本

1. **`extract_masked_files_final.py`** - 最终完整版本
   - 包含完整的蒙版检测逻辑
   - 支持S3和本地文件
   - 生成详细的统计报告

2. **`simple_mask_extractor.py`** - 简化测试版本
   - 仅检测直接蒙版类别
   - 用于验证基本功能

3. **`extract_masked_annotations.py`** - 开发版本
   - 包含更复杂的逻辑
   - 需要完整的perceptron环境

## 使用方法

### 环境准备

```bash
# 安装必要依赖
pip install tqdm numpy

# 如果需要访问S3文件，还需要安装
pip install refile
```

### 运行脚本

```bash
# 运行完整版本
python extract_masked_files_final.py

# 或运行简化测试版本
python simple_mask_extractor.py
```

### 输出结果

脚本会在 `masked_annotations_output` 目录中生成以下文件：

1. **各数据集的蒙版文件列表**：
   - `z10_label_1230_train_masked_files.json`
   - `Z10_label_0207_7w_masked_files.json`
   - `z1_label_1230_bmk_qy_masked_files.json`
   - `BMK_02_masked_files.json`

2. **统计信息**：
   - `statistics.json` - 各数据集的统计摘要
   - `detailed_results.json` - 详细的分析结果

## 输出格式

### 蒙版文件列表格式
```json
[
  "s3://bucket/path/to/file1.json",
  "s3://bucket/path/to/file2.json",
  ...
]
```

### 统计信息格式
```json
{
  "z10_label_1230_train": {
    "total_files": 1000,
    "masked_files": 150,
    "total_frames": 50000,
    "masked_frames": 2500,
    "reason_direct_mask": 100,
    "reason_outlier_box": 30,
    "reason_short_track": 20
  },
  ...
}
```

## 配置说明

脚本中的主要配置参数：

```python
# 数据集配置路径
dataset_configs = {
    "z10_label_1230_train": "s3://wangningzi-data-qy/perceptron_files/0102_z10_labels_1900.json",
    "Z10_label_0207_7w": "s3://gongjiahao-share/e2e/test-file/z10_label_0207_1230jsons.json",
    "z1_label_1230_bmk_qy": "s3://wangningzi-data-qy/perceptron_files/z10_bmk_79.json",
    "BMK_02": "s3://jys-qy/gt_label/linshi/0401/BMK_02.json"
}

# 过滤参数
occlusion_threshold = 1
filter_outlier_boxes = True
filter_short_track = False
soft_occ_threshold = 0.4
```

## 注意事项

1. **网络访问**：如果需要访问S3文件，确保网络连接正常且有相应权限

2. **内存使用**：处理大型数据集时可能需要较多内存

3. **处理时间**：完整处理所有数据集可能需要较长时间

4. **错误处理**：脚本会跳过无法访问或格式错误的文件，并在日志中记录

## 故障排除

### 常见问题

1. **refile模块未找到**
   ```
   pip install refile
   ```

2. **S3访问权限问题**
   - 检查网络连接
   - 确认S3访问权限配置

3. **内存不足**
   - 减少并发处理的文件数量
   - 增加系统内存

4. **文件格式错误**
   - 检查JSON文件格式是否正确
   - 查看错误日志了解具体问题

## 扩展说明

如需修改蒙版检测逻辑，可以调整以下方法：
- `check_direct_mask()` - 直接蒙版类别检测
- `check_outlier_box()` - 异常值检测
- `check_short_track()` - 短轨迹检测

如需添加新的数据集，修改 `dataset_configs` 字典即可。
