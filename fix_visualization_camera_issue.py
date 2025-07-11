#!/usr/bin/env python3
"""
修复前视相机检测框不显示问题的脚本

问题分析:
1. _format_img_size方法中cv2.resize没有赋值回去，导致图像没有真正被resize
2. 图像在draw_boxes_on_imgs之后被resize，但内参矩阵没有相应调整
3. 导致3D到2D投影计算错误，检测框位置不正确

解决方案:
1. 修复_format_img_size方法
2. 在图像resize时同步调整内参矩阵
"""

import os
import shutil

def fix_private_data_loader():
    """修复PrivateDataLoader中的图像处理问题"""
    
    file_path = "dataanalysistool/data_loader/private_data_loader.py"
    backup_path = file_path + ".backup"
    
    # 备份原文件
    if os.path.exists(file_path) and not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"已备份原文件到: {backup_path}")
    
    # 读取原文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复1: _format_img_size方法 - 修复resize不生效的问题
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
                images[camera_id] = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)'''
    
    if old_format_img_size in content:
        content = content.replace(old_format_img_size, new_format_img_size)
        print("✓ 修复了_format_img_size方法中resize不生效的问题")
    else:
        print("⚠ 未找到_format_img_size方法的预期代码，可能已经被修改")
    
    # 修复2: 调整处理顺序，在绘制检测框之前进行图像resize和内参调整
    old_call_sequence = '''            self.draw_boxes_on_imgs(images, self.data["calibrated_sensors"], key_frame)
            if self.draw_cloud:
                self.draw_clouds_on_imgs(images, key_frame)
            self._format_img_size(images)'''
    
    new_call_sequence = '''            # 先进行图像resize，然后调整内参，最后绘制检测框
            self._format_img_size(images)
            self._adjust_intrinsics_for_resized_images(images)
            self.draw_boxes_on_imgs(images, self.data["calibrated_sensors"], key_frame)
            if self.draw_cloud:
                self.draw_clouds_on_imgs(images, key_frame)'''
    
    if old_call_sequence in content:
        content = content.replace(old_call_sequence, new_call_sequence)
        print("✓ 调整了图像处理顺序")
    else:
        print("⚠ 未找到预期的处理顺序代码")
    
    # 添加内参调整方法
    new_method = '''
    def _adjust_intrinsics_for_resized_images(self, images: Dict) -> None:
        """根据图像resize调整内参矩阵"""
        if not hasattr(self, '_original_intrinsics'):
            # 保存原始内参
            self._original_intrinsics = {}
            for camera_id in images.keys():
                if camera_id in self.data["calibrated_sensors"]:
                    original_K = np.array(self.data["calibrated_sensors"][camera_id]["intrinsic"]["K"]).reshape(3, 3)
                    self._original_intrinsics[camera_id] = original_K.copy()
        
        # 计算缩放比例并调整内参
        target_size = self.one_merge_cfg["img_size"]  # [width, height]
        for camera_id, image in images.items():
            if camera_id in self.data["calibrated_sensors"]:
                current_size = list(image.shape[:2])[::-1]  # [width, height]
                
                if current_size != target_size:
                    # 计算缩放比例
                    scale_x = current_size[0] / target_size[0]  # 注意：这里是反向缩放
                    scale_y = current_size[1] / target_size[1]
                    
                    # 调整内参矩阵
                    adjusted_K = self._original_intrinsics[camera_id].copy()
                    adjusted_K[0, 0] /= scale_x  # fx
                    adjusted_K[1, 1] /= scale_y  # fy
                    adjusted_K[0, 2] /= scale_x  # cx
                    adjusted_K[1, 2] /= scale_y  # cy
                    
                    # 更新内参
                    self.data["calibrated_sensors"][camera_id]["intrinsic"]["K"] = adjusted_K.flatten().tolist()
'''
    
    # 在类的末尾添加新方法（在最后一个方法之前）
    last_method_end = content.rfind('        return img\n')
    if last_method_end != -1:
        insert_pos = last_method_end + len('        return img\n')
        content = content[:insert_pos] + new_method + content[insert_pos:]
        print("✓ 添加了内参调整方法")
    else:
        print("⚠ 未找到合适的位置插入新方法")
    
    # 确保导入numpy
    if "import numpy as np" not in content:
        content = content.replace("import numpy", "import numpy as np")
    
    # 写入修复后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 已修复文件: {file_path}")

def fix_visualization_script():
    """修复可视化脚本中的undistort参数"""
    
    file_path = "tools/private_e2e_infer_visualization_single_det.py"
    backup_path = file_path + ".backup"
    
    # 备份原文件
    if os.path.exists(file_path) and not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"已备份原文件到: {backup_path}")
    
    # 读取原文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 启用undistort参数
    old_undistort = '''        # draw_ego_car=True,
        # undistort=True,'''
    
    new_undistort = '''        # draw_ego_car=True,
        undistort=True,  # 启用图像去畸变'''
    
    if old_undistort in content:
        content = content.replace(old_undistort, new_undistort)
        print("✓ 启用了undistort参数")
    else:
        print("⚠ 未找到undistort参数的注释")
    
    # 写入修复后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 已修复文件: {file_path}")

def main():
    """主函数"""
    print("开始修复前视相机检测框不显示问题...")
    print("="*50)
    
    # 修复PrivateDataLoader
    fix_private_data_loader()
    
    print()
    
    # 修复可视化脚本
    fix_visualization_script()
    
    print()
    print("="*50)
    print("修复完成！")
    print()
    print("修复内容:")
    print("1. 修复了_format_img_size方法中cv2.resize不生效的问题")
    print("2. 调整了图像处理顺序：先resize图像，再调整内参，最后绘制检测框")
    print("3. 添加了_adjust_intrinsics_for_resized_images方法来同步调整内参矩阵")
    print("4. 启用了可视化脚本中的undistort参数")
    print()
    print("现在重新运行可视化脚本应该能正确显示前视相机的检测框了。")

if __name__ == "__main__":
    main()
