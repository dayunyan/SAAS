import os
import shutil

# 该文件用于将背景类从auxiliary/full/中提取出来
# 定义文件夹路径
full_folder = "/home/zjj/xjd/datasets/geoeye-1/auxiliary/full/"
residential_folder = "/home/zjj/xjd/datasets/geoeye-1/auxiliary/residential/"
back_folder = "/home/zjj/xjd/datasets/geoeye-1/auxiliary/back/"

# 创建 back 文件夹（如果不存在）
if not os.path.exists(back_folder):
    os.makedirs(back_folder)
else:
    shutil.rmtree(back_folder)
    os.mkdir(back_folder)

residential_list = os.listdir(residential_folder)
# 获取 full 文件夹中的所有图像文件
for root, _, files in os.walk(full_folder):
    for file in files:
        full_image_path = os.path.join(root, file)
        # 检查是否在 residential 文件夹中
        if file not in residential_list:
            # 复制图像到 back 文件夹
            shutil.copy(full_image_path, os.path.join(back_folder, file))

print("已提取并保存非 residential 图像到 back 文件夹。")
