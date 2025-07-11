#!/usr/bin/env python3
"""
验证现有筛选结果中的文件路径是否存在

使用方法:
python validate_existing_results.py <筛选结果文件路径>

例如:
python validate_existing_results.py masked_annotations_output/z10_label_1230_train_masked_files.json
"""

import json
import os
import sys
from tqdm import tqdm

def validate_json_file_paths(json_path):
    """验证单个JSON文件中的数据路径"""
    try:
        # 加载JSON文件
        try:
            import refile
            if json_path.startswith("s3://"):
                with refile.smart_open(json_path, 'r') as f:
                    json_data = json.load(f)
            else:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
        except ImportError:
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
                        "type": "nori_path" if "nori_path" in sensor_info else 
                               "file_path" if "file_path" in sensor_info else "s3_path"
                    })
                    
                    # 只检查前几个缺失文件，避免过慢
                    if len(missing_files) >= 5:
                        break
            
            if len(missing_files) >= 5:
                break
        
        return {
            "valid": len(missing_files) == 0,
            "total_files": total_files,
            "missing_files": missing_files,
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

def validate_masked_files_list(file_list_path):
    """验证蒙版文件列表中的所有JSON文件"""
    
    print(f"验证文件列表: {file_list_path}")
    
    # 加载文件列表
    try:
        with open(file_list_path, 'r', encoding='utf-8') as f:
            json_paths = json.load(f)
    except Exception as e:
        print(f"错误: 无法加载文件列表 - {e}")
        return
    
    print(f"总共 {len(json_paths)} 个JSON文件需要验证")
    
    valid_files = []
    invalid_files = []
    
    # 验证每个JSON文件
    for json_path in tqdm(json_paths, desc="验证JSON文件"):
        result = validate_json_file_paths(json_path)
        
        if result["valid"]:
            valid_files.append(json_path)
        else:
            invalid_files.append({
                "json_path": json_path,
                "error": result.get("error"),
                "missing_files": result["missing_files"],
                "missing_count": result["missing_count"]
            })
    
    # 打印结果
    print("\n" + "="*60)
    print("验证结果")
    print("="*60)
    print(f"总文件数: {len(json_paths)}")
    print(f"有效文件数: {len(valid_files)}")
    print(f"无效文件数: {len(invalid_files)}")
    
    if len(json_paths) > 0:
        valid_ratio = len(valid_files) / len(json_paths) * 100
        print(f"有效文件比例: {valid_ratio:.2f}%")
    
    # 显示一些无效文件的详细信息
    if invalid_files:
        print(f"\n前5个无效文件的详细信息:")
        for i, invalid_file in enumerate(invalid_files[:5]):
            print(f"\n{i+1}. {invalid_file['json_path']}")
            if invalid_file.get('error'):
                print(f"   错误: {invalid_file['error']}")
            elif invalid_file['missing_files']:
                print(f"   缺失文件数: {invalid_file['missing_count']}")
                print(f"   示例缺失文件:")
                for missing in invalid_file['missing_files'][:3]:
                    print(f"     - {missing['sensor']}: {missing['path']}")
    
    # 保存有效文件列表
    if valid_files:
        output_file = file_list_path.replace('.json', '_validated.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(valid_files, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 有效文件列表已保存到: {output_file}")
    
    # 保存验证报告
    report_file = file_list_path.replace('.json', '_validation_report.json')
    report = {
        "summary": {
            "total_files": len(json_paths),
            "valid_files": len(valid_files),
            "invalid_files": len(invalid_files),
            "valid_ratio": len(valid_files) / len(json_paths) * 100 if json_paths else 0
        },
        "invalid_files": invalid_files
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✓ 验证报告已保存到: {report_file}")

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python validate_existing_results.py <筛选结果文件路径>")
        print("\n例如:")
        print("python validate_existing_results.py masked_annotations_output/z10_label_1230_train_masked_files.json")
        sys.exit(1)
    
    file_list_path = sys.argv[1]
    
    if not os.path.exists(file_list_path):
        print(f"错误: 文件不存在 - {file_list_path}")
        sys.exit(1)
    
    validate_masked_files_list(file_list_path)

if __name__ == "__main__":
    main()
