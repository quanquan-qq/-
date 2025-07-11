#!/usr/bin/env python3
"""
简化版本的蒙版提取脚本，用于测试基本功能
"""

import json
import os
import sys
from collections import defaultdict

# 添加perceptron路径
sys.path.append('D:/code')

try:
    from perceptron.data.det3d.source.config import Z10
    print("成功导入Z10配置")
except ImportError as e:
    print(f"导入Z10配置失败: {e}")
    sys.exit(1)

def test_config_loading():
    """测试配置加载"""
    try:
        car_config = Z10()
        print("Z10配置创建成功")
        
        # 检查训练集配置
        if hasattr(car_config, 'trainset_partial'):
            print("训练集配置:")
            for key, value in car_config.trainset_partial.items():
                if key in ["z10_label_1230_train", "Z10_label_0207_7w"]:
                    print(f"  {key}: {value}")
        
        # 检查验证集配置
        if hasattr(car_config, 'benchmark_partial'):
            print("验证集配置:")
            for key, value in car_config.benchmark_partial.items():
                if key in ["z1_label_1230_bmk_qy", "BMK_02"]:
                    print(f"  {key}: {value}")
                    
    except Exception as e:
        print(f"配置加载失败: {e}")
        return False
    
    return True

def test_simple_mask_detection():
    """测试简单的蒙版检测逻辑"""
    # 模拟一些标注数据
    test_annotations = [
        {"category": "蒙版"},
        {"category": "mask"},
        {"category": "小汽车"},
        {"category": "行人"},
        {"category": "masked_area"},
    ]
    
    mask_categories = ["蒙版", "mask", "masked_area", "正向蒙版", "负向蒙版"]
    
    print("测试蒙版检测:")
    for anno in test_annotations:
        category = anno.get("category", "").strip()
        is_mask = category in mask_categories
        print(f"  类别 '{category}': {'是蒙版' if is_mask else '不是蒙版'}")

def main():
    """主测试函数"""
    print("开始测试蒙版提取脚本...")
    
    # 测试配置加载
    if not test_config_loading():
        return
    
    # 测试简单蒙版检测
    test_simple_mask_detection()
    
    print("基本测试完成!")

if __name__ == "__main__":
    main()
