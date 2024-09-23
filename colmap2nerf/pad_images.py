############## 功能：将文件夹中的图片数量补充到200个，并且命名格式顺延（因为val和test图片都没有）############

import os
import shutil

def pad_images_to_200(src_folder):
    # 获取文件夹中的所有图片
    images = [f for f in os.listdir(src_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(images)

    # 如果图片数量已经达到或超过200，则不需要补齐
    if num_images >= 200:
        print(f"图片数量已经有 {num_images} 张，不需要补齐。")
        return

    # 继续复制图片直到达到 200 张
    current_count = num_images
    while current_count < 200:
        # 选择现有的图片来复制
        src_img_name = images[current_count % num_images]  # 循环使用已有图片
        src_img_path = os.path.join(src_folder, src_img_name)

        # 计算新的图片编号，确保顺延命名f"./{str(mode)}/{str(current_count+1).zfill(4)}"
        new_img_name = f"{str(current_count+1).zfill(4)}.png"  # 使用新的顺延命名
        target_img_path = os.path.join(src_folder, new_img_name)

        # 如果文件名已经存在，跳过命名，继续顺延
        if os.path.exists(target_img_path):
            current_count += 1
            continue

        # 复制文件并重命名
        shutil.copy(src_img_path, target_img_path)
        current_count += 1

    print(f"图片已补齐到 200 张，文件保存在 {src_folder}")

# 使用示例
src_folder = '/home/sulutong/mr2nerf-master/colmap2nerf/Real_World_Trans_custom/stump/test'  # 源文件夹路径
pad_images_to_200(src_folder)
