import os
import numpy as np
import glob
from shutil import copy

input_dir = r"D:\data\face_img_50"
file_list = os.listdir(input_dir)
dirs_name = file_list
print(dirs_name)

for name in dirs_name:
    dir_path = os.path.join(input_dir, name)    # 人名目录
    img_list = os.listdir(dir_path)             # 每个人的图像列表
    img_index = np.random.randint(0, len(img_list))     # 随机从每个图片文件夹中各选出一张图片
    select_img_name = img_list[img_index]
    img_path = os.path.join(dir_path, select_img_name)      # 选择出来的图片的路径
    new_path = os.path.join(r"D:\data\faceimg_50_result", name)     # 创建新文件夹保存选择的图片
    os.mkdir(new_path)
    copy(img_path, new_path)




