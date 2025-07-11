#!/usr/bin/env python3
"""
修复文件路径不存在错误的脚本

问题分析:
1. 筛选脚本只验证了JSON文件本身，没有检查JSON中引用的数据文件是否存在
2. 可视化脚本在访问不存在的文件时崩溃
3. 路径转换逻辑可能存在问题

解决方案:
1. 创建增强版的筛选脚本，验证所有引用的数据文件
2. 修改可视化脚本增加错误处理
"""

import json
import os
import sys
from tqdm import tqdm

# 尝试导入refile
try:
    import refile
    HAS_REFILE = True
except ImportError:
    print("警告: refile模块未找到，将使用标准文件操作")
    HAS_REFILE = False

def validate_data_files_in_json(json_path):
    """验证JSON文件中引用的所有数据文件是否存在"""
    try:
        # 加载JSON文件
        if HAS_REFILE and json_path.startswith("s3://"):
            with refile.smart_open(json_path, 'r') as f:
                json_data = json.load(f)
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        
        missing_files = []
        total_files = 0
        
        # 检查每一帧的数据文件
        for frame in json_data.get("frames", []):
            sensor_data = frame.get("sensor_data", {})
            
            for sensor_name, sensor_info in sensor_data.items():
                total_files += 1
                
                # 检查不同类型的文件路径
                file_path = None
                
                if "nori_path" in sensor_info:
                    # nori文件路径
                    nori_path = sensor_info["nori_path"]
                    file_path = nori_path.replace("s3://", "/mnt/acceldata/dynamic/")
                    
                elif "file_path" in sensor_info:
                    # 直接文件路径
                    file_path = sensor_info["file_path"]
                    if file_path.startswith("s3://"):
                        file_path = file_path.replace("s3://", "/mnt/acceldata/dynamic/")
                
                elif "s3_path" in sensor_info:
                    # S3路径
                    s3_path = sensor_info["s3_path"]
                    file_path = s3_path.replace("s3://", "/mnt/acceldata/dynamic/")
                
                # 检查文件是否存在
                if file_path and not os.path.exists(file_path):
                    missing_files.append({
                        "sensor": sensor_name,
                        "path": file_path,
                        "original": sensor_info
                    })
        
        return {
            "valid": len(missing_files) == 0,
            "total_files": total_files,
            "missing_files": missing_files[:10],  # 只记录前10个缺失文件
            "missing_count": len(missing_files)
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "total_files": 0,
            "missing_files": [],
            "missing_count": 0
        }

def create_enhanced_extractor():
    """创建增强版的筛选脚本"""
    
    enhanced_script = '''#!/usr/bin/env python3
"""
增强版蒙版文件提取脚本 - 验证所有数据文件是否存在
"""

import json
import os
from collections import defaultdict
from typing import List, Dict
import numpy as np
from tqdm import tqdm

# 尝试导入refile
try:
    import refile
    HAS_REFILE = True
except ImportError:
    print("警告: refile模块未找到，将使用标准文件操作")
    HAS_REFILE = False

class EnhancedMaskedAnnotationExtractor:
    """增强版蒙版标注提取器 - 验证数据文件存在性"""
    
    def __init__(self):
        # 数据集配置
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
        
        # 结果存储
        self.datasets = {}
        self.valid_files = defaultdict(list)
        self.masked_files = defaultdict(list)
        self.statistics = defaultdict(lambda: defaultdict(int))
    
    def load_json_file(self, file_path):
        """加载JSON文件"""
        try:
            if HAS_REFILE and file_path.startswith("s3://"):
                with refile.smart_open(file_path, 'r') as f:
                    return json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return None
    
    def validate_data_files(self, json_path):
        """验证JSON文件中的数据文件是否存在"""
        try:
            json_data = self.load_json_file(json_path)
            if json_data is None:
                return False, "JSON加载失败"
            
            missing_files = []
            total_files = 0
            
            for frame in json_data.get("frames", []):
                sensor_data = frame.get("sensor_data", {})
                
                for sensor_name, sensor_info in sensor_data.items():
                    total_files += 1
                    
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
                        missing_files.append(file_path)
                        if len(missing_files) > 5:  # 只检查前几个，避免过慢
                            break
                
                if len(missing_files) > 5:
                    break
            
            if missing_files:
                return False, f"缺失文件: {missing_files[0]}"
            
            return True, "所有文件存在"
            
        except Exception as e:
            return False, f"验证失败: {e}"
    
    def check_has_mask(self, json_path):
        """检查JSON文件是否包含蒙版"""
        try:
            json_data = self.load_json_file(json_path)
            if json_data is None:
                return False
            
            for frame in json_data.get("frames", []):
                for annotation in frame.get("labels", []):
                    category = annotation.get("category", "").strip()
                    if category in self.mask_categories:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def load_and_validate_datasets(self):
        """加载并验证数据集"""
        print("开始加载和验证数据集...")
        
        for dataset_name, config_path in self.dataset_configs.items():
            print(f"\\n处理数据集: {dataset_name}")
            
            # 加载路径列表
            json_data = self.load_json_file(config_path)
            if json_data is None:
                print(f"  跳过数据集 {dataset_name}（配置加载失败）")
                continue
            
            if isinstance(json_data, list):
                paths = json_data
            elif isinstance(json_data, dict) and "paths" in json_data:
                paths = json_data["paths"]
            else:
                print(f"  跳过数据集 {dataset_name}（格式不支持）")
                continue
            
            print(f"  总共 {len(paths)} 个JSON文件")
            
            # 验证每个JSON文件
            valid_count = 0
            masked_count = 0
            
            for json_path in tqdm(paths, desc=f"验证 {dataset_name}"):
                # 验证数据文件存在性
                is_valid, reason = self.validate_data_files(json_path)
                
                if is_valid:
                    valid_count += 1
                    self.valid_files[dataset_name].append(json_path)
                    
                    # 检查是否包含蒙版
                    if self.check_has_mask(json_path):
                        masked_count += 1
                        self.masked_files[dataset_name].append(json_path)
                else:
                    # 记录验证失败的原因
                    if "missing_files" not in self.statistics[dataset_name]:
                        self.statistics[dataset_name]["missing_files"] = []
                    self.statistics[dataset_name]["missing_files"].append({
                        "path": json_path,
                        "reason": reason
                    })
            
            print(f"  有效文件: {valid_count}/{len(paths)}")
            print(f"  包含蒙版的有效文件: {masked_count}")
            
            self.statistics[dataset_name]["total"] = len(paths)
            self.statistics[dataset_name]["valid"] = valid_count
            self.statistics[dataset_name]["masked"] = masked_count
    
    def save_results(self, output_dir="enhanced_masked_output"):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\\n保存结果到: {output_dir}")
        
        # 保存有效的蒙版文件列表
        for dataset_name, masked_files in self.masked_files.items():
            if masked_files:
                output_file = os.path.join(output_dir, f"{dataset_name}_valid_masked_files.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(masked_files, f, ensure_ascii=False, indent=2)
                print(f"  {dataset_name}: {len(masked_files)} 个有效蒙版文件")
        
        # 保存统计信息
        stats_file = os.path.join(output_dir, "validation_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.statistics), f, ensure_ascii=False, indent=2)
        
        print(f"统计信息已保存到: {stats_file}")

def main():
    extractor = EnhancedMaskedAnnotationExtractor()
    extractor.load_and_validate_datasets()
    extractor.save_results()
    print("\\n增强版提取完成！现在的文件列表中所有引用的数据文件都存在。")

if __name__ == "__main__":
    main()
'''
    
    with open("extract_masked_files_enhanced.py", 'w', encoding='utf-8') as f:
        f.write(enhanced_script)
    
    print("✓ 已创建增强版筛选脚本: extract_masked_files_enhanced.py")

def fix_visualization_error_handling():
    """修改可视化脚本增加错误处理"""
    
    file_path = "dataanalysistool/data_loader/private_data_loader.py"
    
    if not os.path.exists(file_path):
        print(f"⚠ 文件不存在: {file_path}")
        return
    
    backup_path = file_path + ".backup_paths"
    
    # 备份原文件
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"已备份原文件到: {backup_path}")
    
    # 读取原文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 在_get_images方法中添加错误处理
    old_nori_processing = '''                nori_path = nori_path.replace("s3://", "/mnt/acceldata/dynamic/")
                vreader = nori.nori_reader.VolumesReader(nori_path, [vid], "meta.{0:08x}".format(vid), 2)'''
    
    new_nori_processing = '''                nori_path = nori_path.replace("s3://", "/mnt/acceldata/dynamic/")
                
                # 检查nori路径是否存在
                if not os.path.exists(nori_path):
                    print(f"警告: nori路径不存在: {nori_path}，跳过相机 {k}")
                    continue
                
                vreader = nori.nori_reader.VolumesReader(nori_path, [vid], "meta.{0:08x}".format(vid), 2)'''
    
    if old_nori_processing in content:
        content = content.replace(old_nori_processing, new_nori_processing)
        print("✓ 添加了nori路径存在性检查")
    
    # 在_get_images方法开始添加try-except
    old_get_images_loop = '''        for k in self.camera_list:
            if k not in frame["sensor_data"]:'''
    
    new_get_images_loop = '''        for k in self.camera_list:
            try:
                if k not in frame["sensor_data"]:'''
    
    if old_get_images_loop in content:
        content = content.replace(old_get_images_loop, new_get_images_loop)
        print("✓ 添加了相机处理的try-except")
    
    # 在_get_images方法末尾添加except处理
    old_get_images_end = '''                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                result[k] = img

        return result'''
    
    new_get_images_end = '''                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                result[k] = img
            
            except Exception as e:
                print(f"警告: 处理相机 {k} 时出错: {e}，跳过此相机")
                continue

        return result'''
    
    if old_get_images_end in content:
        content = content.replace(old_get_images_end, new_get_images_end)
        print("✓ 添加了异常处理")
    
    # 确保导入os模块
    if "import os" not in content:
        content = content.replace("import io", "import io\nimport os")
        print("✓ 添加了os模块导入")
    
    # 写入修复后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 已修复文件: {file_path}")

def main():
    """主函数"""
    print("修复文件路径不存在错误")
    print("="*40)
    
    print("\n1. 创建增强版筛选脚本...")
    create_enhanced_extractor()
    
    print("\n2. 修复可视化脚本的错误处理...")
    fix_visualization_error_handling()
    
    print("\n" + "="*40)
    print("修复完成！")
    print("\n使用说明:")
    print("1. 运行增强版筛选脚本:")
    print("   python extract_masked_files_enhanced.py")
    print("\n2. 使用新生成的有效文件列表重新运行可视化:")
    print("   python3 /data/projects/dataanalysistool/tools/private_e2e_infer_visualization_single_det.py \\")
    print("       /path/to/eval_frames.pkl --car-id z10 --save-path /data/outputs/test")
    print("\n增强版筛选脚本会验证所有引用的数据文件是否存在，")
    print("确保生成的文件列表中不包含缺失文件的JSON。")

if __name__ == "__main__":
    main()
