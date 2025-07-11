#!/usr/bin/env python3
"""
脚本用于从指定数据集中提取包含蒙版的JSON文件
根据base.py中的逻辑，蒙版类别是通过以下条件生成的：
1. 遮挡阈值检查失败
2. 异常值过滤检查失败
3. 短轨迹过滤检查失败

作者: AI Assistant
日期: 2025-07-07
"""

import json
import os
import sys
from collections import defaultdict
from typing import List, Dict, Set
import numpy as np
try:
    import refile
except ImportError:
    print("refile模块未找到，使用标准文件操作")
    refile = None
from tqdm import tqdm


class MaskedAnnotationExtractor:
    """提取包含蒙版标注的JSON文件"""
    
    def __init__(self):
        # 数据集配置
        self.datasets = {
            "z10_label_1230_train": [],
            "Z10_label_0207_7w": [],
            "z1_label_1230_bmk_qy": [],
            "BMK_02": []
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
        
        # 遮挡阈值和过滤参数 - 从实验配置复制
        self.occlusion_threshold = 1
        self.filter_outlier_boxes = True
        self.filter_short_track = False
        self.soft_occ_threshold = 0.4
        
        # 结果存储
        self.masked_files = defaultdict(list)
        self.statistics = defaultdict(lambda: defaultdict(int))
        
    def load_dataset_paths(self):
        """加载数据集路径"""
        # 直接使用配置文件中的路径
        dataset_configs = {
            "z10_label_1230_train": "s3://wangningzi-data-qy/perceptron_files/0102_z10_labels_1900.json",
            "Z10_label_0207_7w": "s3://gongjiahao-share/e2e/test-file/z10_label_0207_1230jsons.json",
            "z1_label_1230_bmk_qy": "s3://wangningzi-data-qy/perceptron_files/z10_bmk_79.json",
            "BMK_02": "s3://jys-qy/gt_label/linshi/0401/BMK_02.json"
        }

        for dataset_name, json_path in dataset_configs.items():
            self.datasets[dataset_name] = self._load_json_paths(json_path)

        print("数据集路径加载完成:")
        for name, paths in self.datasets.items():
            print(f"  {name}: {len(paths)} 个JSON文件")
    
    def _load_json_paths(self, json_path):
        """从配置文件加载JSON路径列表"""
        if isinstance(json_path, str):
            try:
                if refile:
                    with refile.smart_open(json_path, 'r') as f:
                        json_data = json.load(f)
                else:
                    # 如果没有refile，尝试本地文件
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)

                if isinstance(json_data, list):
                    return json_data
                elif isinstance(json_data, dict) and "paths" in json_data:
                    return json_data["paths"]
                else:
                    return []
            except Exception as e:
                print(f"加载路径文件失败 {json_path}: {e}")
                return []
        elif isinstance(json_path, list):
            return json_path
        else:
            return []

    def _load_angle_anno(self, label):
        """加载角度标注 - 简化版本"""
        if "yaw_lidar" in label:
            return label["yaw_lidar"]
        elif "rotation" in label and "z" in label["rotation"]:
            return label["rotation"]["z"]
        else:
            return 0.0

    def _load_single_box(self, label):
        """加载单个3D框标注 - 复制自base.py"""
        coor = [
            label["xyz_lidar"]["x"],
            label["xyz_lidar"]["y"],
            label["xyz_lidar"]["z"],
            label["lwh"]["l"],
            label["lwh"]["w"],
            label["lwh"]["h"],
            self._load_angle_anno(label),
        ]
        return np.array(coor, dtype=np.float32)
    
    def _judge_whether_outlier_box(self, box_anno, cat_anno):
        """判断是否为异常值框 - 复制自base.py"""
        if cat_anno not in self.category_map:
            return False
        cur_anno = self.category_map[cat_anno]
        is_outlier = False
        if cur_anno in ["pedestrian"] and (box_anno[3:6] > np.array([3, 3, 3])).any():
            is_outlier = True
        elif cur_anno in ["car", "bus", "bicycle"] and (box_anno[3:6] > np.array([30, 6, 10])).any():
            is_outlier = True
        return is_outlier

    def _get_occlusion_attr(self, anno, camera_keys):
        """获取遮挡属性 - 复制自base.py"""
        if "occlusion" not in anno:
            return True

        occlusion_dict = anno["occlusion"]
        if len(occlusion_dict) == 0:
            return True

        # 检查所有相机的遮挡情况
        for camera_key in camera_keys:
            if camera_key in occlusion_dict:
                if occlusion_dict[camera_key] >= self.soft_occ_threshold:
                    return True
        return False

    def _check_if_masked(self, anno, camera_keys=None):
        """检查标注是否会被标记为蒙版 - 根据base.py逻辑"""
        cat_anno = anno.get("category", "").strip()

        # 直接检查是否已经是蒙版类别
        if cat_anno in ["蒙版", "mask", "masked_area", "正向蒙版", "负向蒙版"]:
            return True, "direct_mask"

        # 检查遮挡阈值
        if self.occlusion_threshold > 0 and camera_keys:
            if not self._get_occlusion_attr(anno, camera_keys):
                return True, "occlusion_threshold"

        # 检查异常值过滤
        if self.filter_outlier_boxes and "xyz_lidar" in anno and "lwh" in anno:
            try:
                box_anno = self._load_single_box(anno)
                if self._judge_whether_outlier_box(box_anno, cat_anno):
                    return True, "outlier_filter"
            except:
                pass

        # 检查短轨迹过滤
        if self.filter_short_track and "track_length" in anno:
            if anno["track_length"] < 2:
                return True, "short_track"

        return False, "normal"

    def check_json_file(self, json_path):
        """检查单个JSON文件是否包含蒙版"""
        try:
            with refile.smart_open(json_path, 'r') as f:
                json_data = json.load(f)

            has_mask = False
            mask_reasons = set()
            total_frames = len(json_data.get("frames", []))
            masked_frames = 0

            # 获取相机键列表
            camera_keys = []
            if "calibrated_sensors" in json_data:
                for sensor_name in json_data["calibrated_sensors"].keys():
                    if sensor_name.startswith("camera"):
                        camera_keys.append(sensor_name)

            for frame in json_data.get("frames", []):
                frame_has_mask = False

                # 检查手标labels
                for label in frame.get("labels", []):
                    is_masked, reason = self._check_if_masked(label, camera_keys)
                    if is_masked:
                        has_mask = True
                        frame_has_mask = True
                        mask_reasons.add(reason)

                # 检查预标pre_labels
                for pre_label in frame.get("pre_labels", []):
                    is_masked, reason = self._check_if_masked(pre_label, camera_keys)
                    if is_masked:
                        has_mask = True
                        frame_has_mask = True
                        mask_reasons.add(reason)

                if frame_has_mask:
                    masked_frames += 1

            return {
                "has_mask": has_mask,
                "mask_reasons": list(mask_reasons),
                "total_frames": total_frames,
                "masked_frames": masked_frames,
                "json_path": json_path
            }

        except Exception as e:
            print(f"处理文件 {json_path} 时出错: {e}")
            return {
                "has_mask": False,
                "mask_reasons": [],
                "total_frames": 0,
                "masked_frames": 0,
                "json_path": json_path,
                "error": str(e)
            }

    def extract_masked_files(self):
        """提取所有包含蒙版的JSON文件"""
        print("开始提取包含蒙版的JSON文件...")

        for dataset_name, json_paths in self.datasets.items():
            if not json_paths:
                print(f"数据集 {dataset_name} 没有找到JSON文件")
                continue

            print(f"\n处理数据集: {dataset_name} ({len(json_paths)} 个文件)")

            for json_path in tqdm(json_paths, desc=f"处理 {dataset_name}"):
                result = self.check_json_file(json_path)

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

        # 为每个数据集保存包含蒙版的JSON文件列表
        for dataset_name, masked_files in self.masked_files.items():
            output_file = os.path.join(output_dir, f"{dataset_name}_masked_files.json")

            # 只保存文件路径列表
            file_paths = [item["json_path"] for item in masked_files]

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(file_paths, f, ensure_ascii=False, indent=2)

            print(f"已保存 {dataset_name}: {len(file_paths)} 个包含蒙版的文件到 {output_file}")

        # 保存详细统计信息
        stats_file = os.path.join(output_dir, "statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.statistics), f, ensure_ascii=False, indent=2)

        # 保存详细结果（包含蒙版原因等信息）
        detailed_file = os.path.join(output_dir, "detailed_results.json")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.masked_files), f, ensure_ascii=False, indent=2)

        print(f"统计信息已保存到 {stats_file}")
        print(f"详细结果已保存到 {detailed_file}")

    def print_summary(self):
        """打印提取结果摘要"""
        print("\n" + "="*60)
        print("蒙版标注提取结果摘要")
        print("="*60)

        for dataset_name in self.datasets.keys():
            stats = self.statistics[dataset_name]
            print(f"\n数据集: {dataset_name}")
            print(f"  总文件数: {stats['total_files']}")
            print(f"  包含蒙版的文件数: {stats['masked_files']}")
            print(f"  蒙版文件比例: {stats['masked_files']/max(stats['total_files'], 1)*100:.2f}%")
            print(f"  总帧数: {stats['total_frames']}")
            print(f"  包含蒙版的帧数: {stats['masked_frames']}")

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


if __name__ == "__main__":
    main()
