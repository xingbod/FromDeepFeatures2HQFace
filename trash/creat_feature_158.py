import os
import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES'] = '-4'
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'# 只显示 Error
import logging
logging.disable(30)# for disable the warnning in gradient tape

from ModelZoo import loadFaceModel, loadArcfaceModel,loadArcfaceModel_xception,loadArcfaceModel_inception
import tqdm
import pickle
import cv2
from tf_utils import allow_memory_growth

allow_memory_growth()
# lfw_dataset_path = "./data/lfw_158"
# names_list = os.listdir(lfw_dataset_path)
# print(names_list)
#
# for name in names_list:
#     imgPair_path = os.path.join("./data/lfw_158", f"{name}")
#     # imgResult_path = os.path.join("./data/celeba_results/result0", f"{name}")
#     if not os.path.exists(imgPair_path):
#         os.mkdir(imgPair_path)
#     # 保存原图
#     source_img_path = os.path.join(imgPair_path, "source_img")
#     if not os.path.exists(source_img_path):
#         os.mkdir(source_img_path)
#         imgs_list = os.listdir(os.path.join(lfw_dataset_path, name))
#         for img in imgs_list:
#             copy(os.path.join(os.path.join(lfw_dataset_path, name), img), source_img_path)





# arcfacemodel = loadArcfaceModel()
arcfacemodel = loadArcfaceModel_inception()
print(arcfacemodel.summary())
lfw_pairs_path = "../data/colorferet_jpg_crop"
names_list = os.listdir(lfw_pairs_path)

lfw_feature_path = "./data/colorferet_158_features_Inception"

for name in tqdm.tqdm(names_list):
    source_img_path = os.path.join(lfw_pairs_path, name)
    # source_img_path = os.path.join(img_path, "source_img")
    source_img = os.listdir(source_img_path)
    feature_path = os.path.join(lfw_feature_path, name)
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)
        source_feature_path = os.path.join(feature_path, "source_feature")
        os.mkdir(source_feature_path)

        source_img_names = os.listdir(source_img_path)
        source_feature = list()
        for inx in range(len(source_img_names)):
            the_source_img = os.path.join(source_img_path, source_img_names[inx])
            # print(the_source_img)
            img = cv2.imread(the_source_img)
            img = cv2.resize(img, (112, 112))
            img = img.astype(np.float32) / 255.
            if len(img.shape) == 3:
                img = np.expand_dims(img, 0)
            feat = arcfacemodel(img)
            # print(feat)
            source_feature.append(feat.numpy())

        file_source = open(f"{source_feature_path}/source.pickle", "wb")
        pickle.dump(source_feature, file_source)
        file_source.close()






