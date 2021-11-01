import os
import glob
from shutil import copy

file_list = os.listdir("D:\data\lfw_mtcnnpy_160\lfw_mtcnnpy_160")
dirs_name = file_list
print(dirs_name)
num = 0
for name in dirs_name:
    dir_path = os.path.join("D:\data\lfw_mtcnnpy_160\lfw_mtcnnpy_160", name)
    img_list = os.listdir(dir_path)
    if len(img_list) >= 10:
        # new_path = os.path.join("D:\data\my_face_images", name)
        # os.mkdir(new_path)
        # for img_name in img_list:
        #     img_path = os.path.join(dir_path, img_name)
        #     copy(img_path, new_path)
        print(img_list)

