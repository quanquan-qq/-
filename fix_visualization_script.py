#!/usr/bin/env python3
"""
修复可视化脚本的补丁文件
使用方法: python fix_visualization_script.py
"""

import os
import shutil

def create_fixed_private_data_loader():
    """创建修复后的private_data_loader.py文件"""
    
    # 备份原文件
    original_file = "dataanalysistool/data_loader/private_data_loader.py"
    backup_file = original_file + ".backup"
    
    if os.path.exists(original_file) and not os.path.exists(backup_file):
        shutil.copy2(original_file, backup_file)
        print(f"已备份原文件到: {backup_file}")
    
    # 读取原文件内容
    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复1: _format_img_size方法
    old_format_img_size = '''    def _format_img_size(self, images: Dict) -> None:
        for image in images.values():
            size = list(image.shape[:2])[::-1]
            if size != self.one_merge_cfg["img_size"]:
                cv2.resize(image, self.one_merge_cfg["img_size"], interpolation=cv2.INTER_AREA)'''
    
    new_format_img_size = '''    def _format_img_size(self, images: Dict) -> None:
        for camera_id, image in images.items():
            size = list(image.shape[:2])[::-1]  # [width, height]
            target_size = self.one_merge_cfg["img_size"]  # [width, height]
            if size != target_size:
                # 修复: 将resize结果赋值回原图像
                images[camera_id] = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                
                # 计算缩放比例并存储，用于后续内参调整
                scale_x = target_size[0] / size[0]
                scale_y = target_size[1] / size[1]
                
                if not hasattr(self, 'image_scales'):
                    self.image_scales = {}
                self.image_scales[camera_id] = (scale_x, scale_y)'''
    
    content = content.replace(old_format_img_size, new_format_img_size)
    
    # 修复2: draw_boxes_on_imgs方法 - 添加内参调整
    old_draw_boxes = '''            intrinsic_k = np.array(sensors_calib[camera_id]["intrinsic"]["K"]).reshape(3, 3)

            images[camera_id] = self._draw_boxes_on_img(camera_id, image, boxes, (transform, intrinsic_k))'''
    
    new_draw_boxes = '''            intrinsic_k = np.array(sensors_calib[camera_id]["intrinsic"]["K"]).reshape(3, 3)
            
            # 如果图像被缩放了，调整内参矩阵
            if hasattr(self, 'image_scales') and camera_id in self.image_scales:
                scale_x, scale_y = self.image_scales[camera_id]
                intrinsic_k[0, 0] *= scale_x  # fx
                intrinsic_k[1, 1] *= scale_y  # fy
                intrinsic_k[0, 2] *= scale_x  # cx
                intrinsic_k[1, 2] *= scale_y  # cy

            images[camera_id] = self._draw_boxes_on_img(camera_id, image, boxes, (transform, intrinsic_k))'''
    
    content = content.replace(old_draw_boxes, new_draw_boxes)
    
    # 修复3: _get_images方法 - 添加错误处理
    old_get_images_start = '''    def _get_images(self, frame) -> np.array:
        """
        Loading image with given sample index

        Args:
            idx (int):, Sampled index
        Returns:
            image (Dict[str, np.ndarray]): (H, W, 3), RGB Image
        """
        result = {}
        for k in self.camera_list:'''
    
    new_get_images_start = '''    def _get_images(self, frame) -> np.array:
        """
        Loading image with given sample index

        Args:
            idx (int):, Sampled index
        Returns:
            image (Dict[str, np.ndarray]): (H, W, 3), RGB Image
        """
        result = {}
        for k in self.camera_list:
            try:'''
    
    content = content.replace(old_get_images_start, new_get_images_start)
    
    # 在_get_images方法末尾添加异常处理
    old_get_images_end = '''                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                result[k] = img

        return result'''
    
    new_get_images_end = '''                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                result[k] = img
            
            except Exception as e:
                print(f"警告: 加载相机 {k} 图像时出错: {e}，跳过此相机")
                continue

        return result'''
    
    content = content.replace(old_get_images_end, new_get_images_end)
    
    # 在nori_path处理中添加路径检查
    old_nori_path = '''                nori_path = nori_path.replace("s3://", "/mnt/acceldata/dynamic/")
                vreader = nori.nori_reader.VolumesReader(nori_path, [vid], "meta.{0:08x}".format(vid), 2)'''
    
    new_nori_path = '''                nori_path = nori_path.replace("s3://", "/mnt/acceldata/dynamic/")
                
                # 检查路径是否存在
                if not os.path.exists(nori_path):
                    print(f"警告: nori路径不存在: {nori_path}，跳过相机 {k}")
                    continue
                    
                vreader = nori.nori_reader.VolumesReader(nori_path, [vid], "meta.{0:08x}".format(vid), 2)'''
    
    content = content.replace(old_nori_path, new_nori_path)
    
    # 添加必要的import
    if "import os" not in content:
        content = content.replace("import io", "import io\nimport os")
    
    # 写入修复后的文件
    with open(original_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已修复文件: {original_file}")
    print("修复内容:")
    print("1. 修复了_format_img_size方法中图像resize不生效的问题")
    print("2. 添加了图像缩放时内参矩阵的相应调整")
    print("3. 增加了文件路径存在性检查和错误处理")

def create_path_validator():
    """创建路径验证脚本"""
    
    validator_content = '''#!/usr/bin/env python3
"""
验证JSON文件中的数据路径是否存在
"""

import json
import os
import refile
from tqdm import tqdm

def validate_single_json(json_path):
    """验证单个JSON文件中的所有数据路径"""
    try:
        if json_path.startswith("s3://"):
            with refile.smart_open(json_path, 'r') as f:
                json_data = json.load(f)
        else:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
        
        missing_files = []
        
        for frame in json_data.get("frames", []):
            sensor_data = frame.get("sensor_data", {})
            
            for sensor_name, sensor_info in sensor_data.items():
                # 检查图像文件
                if "nori_path" in sensor_info:
                    nori_path = sensor_info["nori_path"]
                    local_path = nori_path.replace("s3://", "/mnt/acceldata/dynamic/")
                    if not os.path.exists(local_path):
                        missing_files.append(local_path)
                
                elif "file_path" in sensor_info:
                    file_path = sensor_info["file_path"]
                    if file_path.startswith("s3://"):
                        local_path = file_path.replace("s3://", "/mnt/acceldata/dynamic/")
                        if not os.path.exists(local_path):
                            missing_files.append(local_path)
                
                elif "s3_path" in sensor_info:
                    s3_path = sensor_info["s3_path"]
                    local_path = s3_path.replace("s3://", "/mnt/acceldata/dynamic/")
                    if not os.path.exists(local_path):
                        missing_files.append(local_path)
        
        return len(missing_files) == 0, missing_files
        
    except Exception as e:
        return False, [f"JSON parsing error: {e}"]

def filter_valid_json_files(input_json_list, output_json_list):
    """过滤出所有数据文件都存在的JSON文件"""
    
    with open(input_json_list, 'r') as f:
        json_paths = json.load(f)
    
    valid_paths = []
    invalid_paths = []
    
    for json_path in tqdm(json_paths, desc="验证JSON文件"):
        is_valid, missing_files = validate_single_json(json_path)
        
        if is_valid:
            valid_paths.append(json_path)
        else:
            invalid_paths.append({
                "json_path": json_path,
                "missing_files": missing_files[:5]  # 只记录前5个缺失文件
            })
    
    # 保存有效路径
    with open(output_json_list, 'w') as f:
        json.dump(valid_paths, f, indent=2)
    
    # 保存无效路径报告
    invalid_report = output_json_list.replace('.json', '_invalid_report.json')
    with open(invalid_report, 'w') as f:
        json.dump(invalid_paths, f, indent=2)
    
    print(f"总计: {len(json_paths)} 个文件")
    print(f"有效: {len(valid_paths)} 个文件")
    print(f"无效: {len(invalid_paths)} 个文件")
    print(f"有效文件列表保存到: {output_json_list}")
    print(f"无效文件报告保存到: {invalid_report}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("用法: python validate_json_paths.py <input_json_list> <output_json_list>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    filter_valid_json_files(input_file, output_file)
'''
    
    with open("validate_json_paths.py", 'w', encoding='utf-8') as f:
        f.write(validator_content)
    
    print("已创建路径验证脚本: validate_json_paths.py")

def main():
    """主函数"""
    print("开始修复可视化脚本...")
    
    # 修复private_data_loader.py
    create_fixed_private_data_loader()
    
    # 创建路径验证脚本
    create_path_validator()
    
    print("\n修复完成！")
    print("\n使用说明:")
    print("1. 首先验证JSON文件路径:")
    print("   python validate_json_paths.py your_masked_files.json valid_masked_files.json")
    print("\n2. 然后使用有效的JSON文件列表重新运行可视化:")
    print("   python3 /data/projects/dataanalysistool/tools/private_e2e_infer_visualization_single_det.py \\")
    print("       /path/to/eval_frames.pkl --car-id z10 --save-path /data/outputs/test")

if __name__ == "__main__":
    main()
