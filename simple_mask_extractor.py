#!/usr/bin/env python3
"""
简化版蒙版提取脚本，用于测试基本功能
"""

import json
import os
from collections import defaultdict

def check_direct_mask_categories(annotation):
    """检查是否直接包含蒙版类别"""
    category = annotation.get("category", "").strip()
    mask_categories = ["蒙版", "mask", "masked_area", "正向蒙版", "负向蒙版"]
    return category in mask_categories

def analyze_single_json(json_path):
    """分析单个JSON文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        has_mask = False
        total_frames = len(json_data.get("frames", []))
        masked_frames = 0
        mask_count = 0
        
        for frame in json_data.get("frames", []):
            frame_has_mask = False
            
            # 检查手标labels
            for label in frame.get("labels", []):
                if check_direct_mask_categories(label):
                    has_mask = True
                    frame_has_mask = True
                    mask_count += 1
            
            # 检查预标pre_labels
            for pre_label in frame.get("pre_labels", []):
                if check_direct_mask_categories(pre_label):
                    has_mask = True
                    frame_has_mask = True
                    mask_count += 1
            
            if frame_has_mask:
                masked_frames += 1
        
        return {
            "has_mask": has_mask,
            "total_frames": total_frames,
            "masked_frames": masked_frames,
            "mask_count": mask_count,
            "json_path": json_path
        }
        
    except Exception as e:
        print(f"处理文件 {json_path} 时出错: {e}")
        return None

def test_with_sample_data():
    """使用示例数据测试"""
    # 创建示例JSON数据
    sample_data = {
        "frames": [
            {
                "labels": [
                    {"category": "小汽车", "xyz_lidar": {"x": 1, "y": 2, "z": 3}},
                    {"category": "蒙版", "xyz_lidar": {"x": 4, "y": 5, "z": 6}}
                ],
                "pre_labels": [
                    {"category": "行人", "xyz_lidar": {"x": 7, "y": 8, "z": 9}}
                ]
            },
            {
                "labels": [
                    {"category": "货车", "xyz_lidar": {"x": 10, "y": 11, "z": 12}}
                ],
                "pre_labels": [
                    {"category": "mask", "xyz_lidar": {"x": 13, "y": 14, "z": 15}},
                    {"category": "自行车", "xyz_lidar": {"x": 16, "y": 17, "z": 18}}
                ]
            }
        ]
    }
    
    # 保存示例数据
    sample_file = "sample_data.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    # 分析示例数据
    result = analyze_single_json(sample_file)
    
    print("示例数据分析结果:")
    print(f"  包含蒙版: {result['has_mask']}")
    print(f"  总帧数: {result['total_frames']}")
    print(f"  包含蒙版的帧数: {result['masked_frames']}")
    print(f"  蒙版标注总数: {result['mask_count']}")
    
    # 清理示例文件
    os.remove(sample_file)
    
    return result

def main():
    """主函数"""
    print("开始测试简化版蒙版提取脚本...")
    
    # 测试基本功能
    result = test_with_sample_data()
    
    if result and result['has_mask']:
        print("✓ 基本蒙版检测功能正常")
    else:
        print("✗ 基本蒙版检测功能异常")
    
    print("\n如果基本功能正常，可以继续使用完整版脚本处理实际数据")

if __name__ == "__main__":
    main()
