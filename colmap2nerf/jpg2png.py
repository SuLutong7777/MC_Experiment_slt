######################## 功能：将文件夹中的jpg格式图片全部转换为png格式 #########################

import os
from PIL import Image

def convert_and_delete_jpg(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):  # 检查是否为JPG文件（忽略大小写）
            # 构造文件的完整路径
            jpg_path = os.path.join(folder_path, filename)
            # 打开JPG文件
            img = Image.open(jpg_path)
            # 构造新的PNG文件路径
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(folder_path, png_filename)
            # 保存为PNG格式
            img.save(png_path, 'PNG')
            print(f"转换 {jpg_path} 为 {png_path}")

            # 删除原始的JPG文件
            os.remove(jpg_path)
            print(f"已删除 {jpg_path}")

# 使用示例：调用该函数并传递包含JPG文件的文件夹路径
folder_path = "/home/sulutong/mr2nerf-master/colmap2nerf/Real_World_Trans/bonsai/images_8"  # 替换为实际的文件夹路径
convert_and_delete_jpg(folder_path)
