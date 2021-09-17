import os
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from shutil import copy
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'# 只显示 Error
import logging
logging.disable(30)# for disable the warnning in gradient tape

from ModelZoo import loadStyleGAN2Model, loadArcfaceModel

from skimage import io
from skimage.transform import resize
import pickle


# lfw_dataset_path = r"D:\data\celeba_select"
# names_list = os.listdir(lfw_dataset_path)
# print(names_list)
#
# for name in names_list:
#     imgPair_path = os.path.join(r"D:\data\celeba_pairs", f"{name}_pair")
#     imgResult_path = os.path.join(r"D:\data\celeba_select_result", f"{name}")
#     if not os.path.exists(imgPair_path):
#         os.mkdir(imgPair_path)
#     # 保存原图
#     source_img_path = os.path.join(imgPair_path, "source_img")
#     if not os.path.exists(source_img_path):
#         os.mkdir(source_img_path)
#         imgs_list = os.listdir(os.path.join(lfw_dataset_path, name))
#         for img in imgs_list:
#             copy(os.path.join(os.path.join(lfw_dataset_path, name), img), source_img_path)
#     # 保存ground truth
#     img_result_list = os.listdir(imgResult_path)
#     gt_img = [idx for idx in img_result_list if idx[0:2] == 'gt']
#     gt_img_path = os.path.join(imgPair_path, "gt_img")
#     if not os.path.exists(gt_img_path):
#         os.mkdir(gt_img_path)
#         copy(os.path.join(imgResult_path, gt_img[0]), gt_img_path)
#    # 保存预测图
#     pred_img_path = os.path.join(imgPair_path, "pred_img")
#     if not os.path.exists(pred_img_path):
#         os.mkdir(pred_img_path)
#         out_imgs = [idx for idx in img_result_list if idx[0:3] == 'out']
#         loss_array = list()
#         for out_imgs_name in out_imgs:
#             (filename, extension) = os.path.splitext(out_imgs_name)
#             name_split = filename.split('_', )
#             loss_array.append(float(name_split[-1]))
#         index = int(np.argmin(loss_array))
#         copy(os.path.join(imgResult_path, out_imgs[index]), pred_img_path)




arcfacemodel = loadArcfaceModel()

lfw_pairs_path = "./data/celeba_pairs"
names_list = os.listdir(lfw_pairs_path)

lfw_feature_path = "./data/celeba_feature_pairs"

for name in names_list:
    img_path = os.path.join(lfw_pairs_path, name)
    gt_img_path = os.path.join(img_path, "gt_img")
    pred_img_path = os.path.join(img_path, "pred_img")
    source_img_path = os.path.join(img_path, "source_img")
    gt_img = os.listdir(gt_img_path)
    pred_img = os.listdir(pred_img_path)
    source_img = os.listdir(source_img_path)
    feature_path = os.path.join(lfw_feature_path, name)
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)
        gt_feature_path = os.path.join(feature_path, "gt_feature")
        pred_feature_path = os.path.join(feature_path, "pred_feature")
        source_feature_path = os.path.join(feature_path, "source_feature")
        os.mkdir(gt_feature_path)
        os.mkdir(pred_feature_path)
        os.mkdir(source_feature_path)

        gt_img_name = os.listdir(gt_img_path)
        pred_img_name = os.listdir(pred_img_path)
        source_img_names = os.listdir(source_img_path)

        the_gt_img = os.path.join(gt_img_path, gt_img_name[0])
        img = io.imread(the_gt_img)
        img = resize(img, (112, 112), anti_aliasing=True)
        img = np.array(img)
        img = np.expand_dims(img, 0)
        gt_feature = arcfacemodel(img)
        file_gt = open(f"{gt_feature_path}/gt.pickle", "wb")
        pickle.dump(gt_feature.numpy(), file_gt)
        file_gt.close()


        the_pred_img = os.path.join(pred_img_path, pred_img_name[0])
        img2 = io.imread(the_pred_img)
        img2 = resize(img2, (112, 112), anti_aliasing=True)
        img2 = np.array(img2)
        img2 = np.expand_dims(img2, 0)
        pred_feature = arcfacemodel(img2)
        file_pred = open(f"{pred_feature_path}/pred.pickle", "wb")
        pickle.dump(pred_feature.numpy(), file_pred)
        file_pred.close()

        source_feature = list()
        for inx in range(len(source_img_names)):
            the_source_img = os.path.join(source_img_path, source_img_names[inx])
            img = io.imread(the_source_img)
            img = resize(img, (112, 112), anti_aliasing=True)
            img = np.array(img)
            img = np.expand_dims(img, 0)
            source_feature.append(arcfacemodel(img).numpy())

        file_source = open(f"{source_feature_path}/source.pickle", "wb")
        pickle.dump(source_feature, file_source)
        file_source.close()




