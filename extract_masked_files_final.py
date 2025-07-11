#!/usr/bin/env python3
"""
最终版本：从指定数据集中提取包含蒙版的JSON文件

使用方法:
1. 确保已安装必要的依赖: pip install tqdm numpy refile
2. 运行脚本: python extract_masked_files_final.py
3. 结果将保存在 masked_annotations_output 目录中

根据base.py中的逻辑，蒙版类别通过以下条件生成：
1. 直接标注为蒙版类别（"蒙版", "mask", "masked_area"等）
2. 遮挡阈值检查失败
3. 异常值过滤检查失败  
4. 短轨迹过滤检查失败

"""

import json
import os
from collections import defaultdict
from typing import List, Dict
import numpy as np
from tqdm import tqdm

# 尝试导入refile，如果失败则使用标准文件操作
try:
    import refile
    HAS_REFILE = True
except ImportError:
    print("警告: refile模块未找到，将使用标准文件操作")
    HAS_REFILE = False


class MaskedAnnotationExtractor:
    """提取包含蒙版标注的JSON文件"""
    
    def __init__(self):
        # 数据集配置 - 直接从z10.py复制
        self.dataset_configs = {
            "z10_label_1230_train": "s3://wangningzi-data-qy/perceptron_files/0102_z10_labels_1900.json",
            "Z10_label_0207_7w": "s3://gongjiahao-share/e2e/test-file/z10_label_0207_1230jsons.json",
            "z1_label_1230_bmk_qy": "s3://wangningzi-data-qy/perceptron_files/z10_bmk_79.json",
            "BMK_02": "s3://jys-qy/gt_label/linshi/0401/BMK_02.json"
        }
        
        # 蒙版类别列表
        self.mask_categories = {
            "蒙版", "mask", "masked_area", "正向蒙版", "负向蒙版"
        }
        
        # 类别映射 - 从配置文件复制
        self.category_map = {
            "小汽车": "car", "汽车": "car", "货车": "truck", "工程车": "construction_vehicle",
            "巴士": "bus", "摩托车": "motorcycle", "自行车": "bicycle", "三轮车": "tricycle",
            "骑车人": "cyclist", "骑行的人": "cyclist", "人": "pedestrian", "行人": "pedestrian",
            "其它": "other", "其他": "other", "残影": "ghost", "蒙版": "masked_area",
            "suv": "car", "SUV": "car", "van": "car", "VAN": "car", "Van": "car",
            "皮卡": "car", "pika": "car", "cart": "car", "car": "car", "truck": "truck",
            "construction_vehicle": "construction_vehicle", "bus": "bus",
            "motorcycle": "motorcycle", "bicycle": "bicycle", "tricycle": "tricycle",
            "cyclist": "cyclist", "pedestrian": "pedestrian", "other": "other",
            "ghost": "ghost", "masked_area": "masked_area", "遮挡": "occlusion",
            "短障碍物": "short_track", "大货车": "truck", "dahuoche": "truck",
            "dauhoche": "truck", "小货车": "truck", "xiaohuoche": "truck",
            "骑三轮车的人": "tricycle", "骑自行车的人": "cyclist", "骑摩托车的人": "cyclist",
            "et": "pedestrian", "儿童": "pedestrian", "成年人": "pedestrian",
            "蒙版": "masked_area", "mask": "masked_area", "正向蒙版": "masked_area",
            "负向蒙版": "masked_area", "拖挂": "truck", "tuogua": "truck",
            "其他非机动车": "other", "其他机动车": "other", "小动物类": "other", "大动物类": "other"
        }
        
        # 过滤参数
        self.occlusion_threshold = 1
        self.filter_outlier_boxes = True
        self.filter_short_track = False
        self.soft_occ_threshold = 0.4
        
        # 结果存储
        self.datasets = {}
        self.masked_files = defaultdict(list)
        self.statistics = defaultdict(lambda: defaultdict(int))
    
    def load_json_file(self, file_path):
        """加载JSON文件"""
        try:
            if HAS_REFILE and file_path.startswith("s3://"):
                with refile.smart_open(file_path, 'r') as f:
                    return json.load(f)
            else:
                # 本地文件或没有refile
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return None
    
    def load_dataset_paths(self):
        """加载数据集路径"""
        print("开始加载数据集路径...")
        
        for dataset_name, config_path in self.dataset_configs.items():
            print(f"加载数据集: {dataset_name}")
            json_data = self.load_json_file(config_path)
            
            if json_data is None:
                print(f"  跳过数据集 {dataset_name}（加载失败）")
                self.datasets[dataset_name] = []
                continue
            
            # 解析路径列表
            if isinstance(json_data, list):
                paths = json_data
            elif isinstance(json_data, dict) and "paths" in json_data:
                paths = json_data["paths"]
            else:
                print(f"  跳过数据集 {dataset_name}（格式不支持）")
                paths = []
            
            self.datasets[dataset_name] = paths
            print(f"  加载完成: {len(paths)} 个JSON文件")
        
        print("数据集路径加载完成")
    
    def check_direct_mask(self, annotation):
        """检查是否直接包含蒙版类别"""
        category = annotation.get("category", "").strip()
        return category in self.mask_categories
    
    def check_outlier_box(self, annotation):
        """检查是否为异常值框"""
        if not self.filter_outlier_boxes:
            return False
        
        category = annotation.get("category", "").strip()
        if category not in self.category_map:
            return False
        
        # 检查是否有3D框信息
        if not all(key in annotation for key in ["xyz_lidar", "lwh"]):
            return False
        
        try:
            # 获取框尺寸
            l = annotation["lwh"]["l"]
            w = annotation["lwh"]["w"] 
            h = annotation["lwh"]["h"]
            size = np.array([l, w, h])
            
            mapped_category = self.category_map[category]
            
            # 异常值检查规则
            if mapped_category == "pedestrian" and (size > np.array([3, 3, 3])).any():
                return True
            elif mapped_category in ["car", "bus", "bicycle"] and (size > np.array([30, 6, 10])).any():
                return True
                
        except (KeyError, TypeError, ValueError):
            pass
        
        return False
    
    def check_short_track(self, annotation):
        """检查是否为短轨迹"""
        if not self.filter_short_track:
            return False
        return annotation.get("track_length", 2) < 2
    
    def validate_data_files(self, json_path):
        """验证JSON文件中的数据文件是否存在"""
        try:
            json_data = self.load_json_file(json_path)
            if json_data is None:
                return False, "JSON加载失败"

            # 检查前几帧的数据文件
            for frame_idx, frame in enumerate(json_data.get("frames", [])[:3]):  # 只检查前3帧
                sensor_data = frame.get("sensor_data", {})

                for sensor_name, sensor_info in sensor_data.items():
                    file_path = None
                    if "nori_path" in sensor_info:
                        nori_path = sensor_info["nori_path"]
                        file_path = nori_path.replace("s3://", "/mnt/acceldata/dynamic/")
                    elif "file_path" in sensor_info:
                        file_path = sensor_info["file_path"]
                        if file_path.startswith("s3://"):
                            file_path = file_path.replace("s3://", "/mnt/acceldata/dynamic/")
                    elif "s3_path" in sensor_info:
                        s3_path = sensor_info["s3_path"]
                        file_path = s3_path.replace("s3://", "/mnt/acceldata/dynamic/")

                    if file_path and not os.path.exists(file_path):
                        return False, f"文件不存在: {file_path}"

            return True, "文件验证通过"

        except Exception as e:
            return False, f"验证失败: {e}"

    def analyze_json_file(self, json_path):
        """分析单个JSON文件"""
        # 首先验证数据文件是否存在
        is_valid, reason = self.validate_data_files(json_path)
        if not is_valid:
            print(f"跳过文件 {json_path}: {reason}")
            return None

        json_data = self.load_json_file(json_path)
        if json_data is None:
            return None
        
        has_mask = False
        mask_reasons = set()
        total_frames = len(json_data.get("frames", []))
        masked_frames = 0
        
        for frame in json_data.get("frames", []):
            frame_has_mask = False
            
            # 检查所有标注（labels和pre_labels）
            all_annotations = []
            all_annotations.extend(frame.get("labels", []))
            all_annotations.extend(frame.get("pre_labels", []))
            
            for annotation in all_annotations:
                # 检查各种蒙版条件
                if self.check_direct_mask(annotation):
                    has_mask = True
                    frame_has_mask = True
                    mask_reasons.add("direct_mask")
                elif self.check_outlier_box(annotation):
                    has_mask = True
                    frame_has_mask = True
                    mask_reasons.add("outlier_box")
                elif self.check_short_track(annotation):
                    has_mask = True
                    frame_has_mask = True
                    mask_reasons.add("short_track")
            
            if frame_has_mask:
                masked_frames += 1
        
        return {
            "has_mask": has_mask,
            "mask_reasons": list(mask_reasons),
            "total_frames": total_frames,
            "masked_frames": masked_frames,
            "json_path": json_path
        }
    
    def extract_masked_files(self):
        """提取所有包含蒙版的JSON文件"""
        print("\n开始提取包含蒙版的JSON文件...")
        
        for dataset_name, json_paths in self.datasets.items():
            if not json_paths:
                print(f"数据集 {dataset_name} 没有JSON文件")
                continue
            
            print(f"\n处理数据集: {dataset_name} ({len(json_paths)} 个文件)")
            
            for json_path in tqdm(json_paths, desc=f"处理 {dataset_name}"):
                result = self.analyze_json_file(json_path)
                
                if result is None:
                    continue
                
                # 更新统计信息
                self.statistics[dataset_name]["total_files"] += 1
                self.statistics[dataset_name]["total_frames"] += result["total_frames"]
                
                if result["has_mask"]:
                    self.masked_files[dataset_name].append(result)
                    self.statistics[dataset_name]["masked_files"] += 1
                    self.statistics[dataset_name]["masked_frames"] += result["masked_frames"]
                    
                    # 统计蒙版原因
                    for reason in result["mask_reasons"]:
                        self.statistics[dataset_name][f"reason_{reason}"] += 1
    
    def save_results(self, output_dir="masked_annotations_output"):
        """保存结果到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n保存结果到目录: {output_dir}")
        
        # 为每个数据集保存包含蒙版的JSON文件列表
        for dataset_name, masked_files in self.masked_files.items():
            output_file = os.path.join(output_dir, f"{dataset_name}_masked_files.json")
            
            # 只保存文件路径列表
            file_paths = [item["json_path"] for item in masked_files]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(file_paths, f, ensure_ascii=False, indent=2)
            
            print(f"已保存 {dataset_name}: {len(file_paths)} 个包含蒙版的文件")
        
        # 保存统计信息
        stats_file = os.path.join(output_dir, "statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.statistics), f, ensure_ascii=False, indent=2)
        
        # 保存详细结果
        detailed_file = os.path.join(output_dir, "detailed_results.json")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.masked_files), f, ensure_ascii=False, indent=2)
        
        print(f"统计信息已保存到: {stats_file}")
        print(f"详细结果已保存到: {detailed_file}")
    
    def print_summary(self):
        """打印提取结果摘要"""
        print("\n" + "="*60)
        print("蒙版标注提取结果摘要")
        print("="*60)
        
        for dataset_name in self.dataset_configs.keys():
            stats = self.statistics[dataset_name]
            print(f"\n数据集: {dataset_name}")
            print(f"  总文件数: {stats.get('total_files', 0)}")
            print(f"  包含蒙版的文件数: {stats.get('masked_files', 0)}")
            
            total_files = stats.get('total_files', 0)
            if total_files > 0:
                ratio = stats.get('masked_files', 0) / total_files * 100
                print(f"  蒙版文件比例: {ratio:.2f}%")
            
            print(f"  总帧数: {stats.get('total_frames', 0)}")
            print(f"  包含蒙版的帧数: {stats.get('masked_frames', 0)}")
            
            # 打印蒙版原因统计
            reasons = [k for k in stats.keys() if k.startswith("reason_")]
            if reasons:
                print("  蒙版原因统计:")
                for reason_key in reasons:
                    reason = reason_key.replace("reason_", "")
                    count = stats[reason_key]
                    print(f"    {reason}: {count} 个文件")


def main():
    """主函数"""
    print("蒙版标注提取脚本")
    print("="*40)
    
    extractor = MaskedAnnotationExtractor()
    
    # 加载数据集路径
    extractor.load_dataset_paths()
    
    # 提取包含蒙版的文件
    extractor.extract_masked_files()
    
    # 打印摘要
    extractor.print_summary()
    
    # 保存结果
    extractor.save_results()
    
    print("\n提取完成！")
    print("结果已保存在 masked_annotations_output 目录中")


if __name__ == "__main__":
    main()
