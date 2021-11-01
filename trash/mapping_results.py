import os
import pickle

import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from shutil import copy

import matplotlib.pyplot as plt
import itertools
# import cv2
import seaborn as sns
from pyeer.eer_info import get_eer_stats

from tf_utils import allow_memory_growth

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
logging.disable(30)# for disable the warnning in gradient tape
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow.keras import Model, optimizers, layers, losses
from ModelZoo import loadStyleGAN2Model, loadArcfaceModel,mytestModel
from PIL import Image
from stylegan2.utils import postprocess_images
import random
from tqdm import tqdm

allow_memory_growth()

# arcfacemodel = loadArcfaceModel()
# g_clone = loadStyleGAN2Model()
# model = tf.saved_model.load('./data/rep_try/models/batch21000')
#
#
# name_list = os.listdir("./data/lfw_158")
#
# result_path = './data/mapping_results/lfw_158_result'
# if not os.path.exists(result_path):
#     os.mkdir(result_path)
#
# for name in name_list:
#     img_path = os.path.join("./data/lfw_158", name)
#     img_save_path = os.path.join(result_path, name)
#     if not os.path.exists(img_save_path):
#         os.mkdir(img_save_path)
#     img_name_list = os.listdir(img_path)
#     gt_img_name = random.sample(img_name_list, 1)
#     gt_img_path = os.path.join(img_path, gt_img_name[0])
#     img = io.imread(gt_img_path)
#     gt_img = np.array(img)
#     Image.fromarray(gt_img, 'RGB').save(img_save_path + r'/gt_' + name + '.png')
#     img = resize(img, (112, 112), anti_aliasing=True)
#     img = np.array(img)
#     img = np.expand_dims(img, 0)
#     gt_feature = arcfacemodel(img)
#     with open(f'./data/mapping_results/gt_{name}.pickle', 'wb') as f:
#         pickle.dump(gt_feature.numpy(), f)
#
#     z_pred = model(np.expand_dims(gt_feature.numpy(), 0))
#     img_out = g_clone([z_pred, []], training=False, truncation_psi=0.5)
#     img_out = postprocess_images(img_out)
#     feture_out = arcfacemodel(tf.image.resize(img_out, size=(112, 112)) / 255.)
#     img_out = tf.cast(img_out, dtype=tf.dtypes.uint8)
#     img_out = img_out.numpy()
#     Image.fromarray(img_out[0], 'RGB').save(img_save_path + r'/img_out_' + name + '.png')
#     with open(f'./data/mapping_results/img_out_{name}.pickle', 'wb') as f:
#         pickle.dump(feture_out.numpy(), f)




def euc_sim(a, b):
    return 1 - np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b))


# features_list = list()
def nchoosek(startnum, endnum, step=1, n=1):
    c = []
    for i in itertools.combinations(range(startnum,endnum+1,step),n):
        c.append(list(i))
    return c


# calculate imposter scores
def cal_imposter_scorces():
    features_path = "./data/lfw_158_features_Res50"
    names_list = os.listdir(features_path)
    imposter_scores = []
    for i,j in nchoosek(0, 157, step=1, n=2):
        with open(os.path.join(features_path, names_list[i], 'source_feature', 'source.pickle'), 'rb') as file:
            features1 = np.array(pickle.load(file))[:3]
        with open(os.path.join(features_path, names_list[j], 'source_feature', 'source.pickle'), 'rb') as file:
            features2 = np.array(pickle.load(file))[:3]
        imposter_scores.append(euc_sim(features1 , features2))
    imp = open(f"./data/scores/lfw_mapping_imposter_Res50.pickle", "wb")
    pickle.dump(imposter_scores, imp)
    imp.close()
    # print(imposter_scores[:20])
    return imposter_scores


def cal_genuine_scorces():
    features_path = "./data/lfw_158_features_Res50"
    names_list = os.listdir(features_path)
    genuine_scores = []
    for name in names_list:
        with open(os.path.join(features_path, name, 'source_feature', 'source.pickle'), 'rb') as file:
            features_list = np.array(pickle.load(file))
        for i, j in nchoosek(0, len(features_list)-1, step=1, n=2):
            feature1 = features_list[i]
            feature2 = features_list[j]
            genuine_scores.append(euc_sim(feature1, feature2))
    gen = open(f"./data/scores/lfw_mapping_genuine_Res50.pickle", "wb")
    pickle.dump(genuine_scores, gen)
    gen.close()
    # print(genuine_scores[:20])
    return genuine_scores


def cal_attack_scorces():
    attack_scores = []
    features_path = "./data/mapping_results/lfw_158_result"
    names_list = os.listdir(features_path)
    for name in names_list:
        with open(f'./data/mapping_results/gt_{name}.pickle', 'rb') as gt_file:
            gt_feature = np.array(pickle.load(gt_file))
        with open(f'./data/mapping_results/img_out_{name}.pickle', 'rb') as pred_file:
            pred_feature = np.array(pickle.load(pred_file))
        attack_scores.append(euc_sim(gt_feature, pred_feature))

    att1 = open(f"./data/scores/lfw_mapping_attack1_Res50.pickle", "wb")
    pickle.dump(attack_scores, att1)
    att1.close()
    return attack_scores


def cal_attack_scorces2():
    features_path = "./data/mapping_results/lfw_158_result"
    names_list = os.listdir(features_path)
    attack_scores2 = []
    for name in names_list:
        with open(f'./data/mapping_results/img_out_{name}.pickle', 'rb') as pred_file:
            pred_feature = np.array(pickle.load(pred_file))
        with open(f'./data/lfw_158_features_Res50/{name}/source_feature/source.pickle', 'rb') as file:
            features_list = np.array(pickle.load(file))
        for i in range(len(features_list)):
            attack_scores2.append(euc_sim(pred_feature, features_list[i]))
    att2 = open(f"./data/scores/lfw_mapping_attack2_Res50.pickle", "wb")
    pickle.dump(attack_scores2, att2)
    att2.close()
    return attack_scores2


def draw_pic1():
    sns.set_style("white")
    kwargs = dict(kde_kws={'linewidth': 0.001})
    plt.figure(figsize=(10, 7), dpi=80)
    gen = cal_genuine_scorces()
    imp = cal_imposter_scorces()
    sns.distplot(gen, color="dodgerblue", label="Genuine score", **kwargs)
    sns.distplot(imp, color="orange", label="Imposter score", **kwargs)
    sns.distplot(cal_attack_scorces(), color="deeppink", label="Reconstructed face score", **kwargs)
    plt.legend()
    plt.savefig("lfw_Res50_mapping_type1.svg")
    stats_a = get_eer_stats(gen, imp)
    # print(stats_a)
    print('stats_a.eer',stats_a.eer)# 10%

def draw_pic2():
    sns.set_style("white")
    kwargs = dict(kde_kws={'linewidth': 0.001})
    plt.figure(figsize=(10, 7), dpi=80)
    sns.distplot(cal_genuine_scorces(), color="dodgerblue", label="Genuine score", **kwargs)
    sns.distplot(cal_imposter_scorces(), color="orange", label="Imposter score", **kwargs)
    sns.distplot(cal_attack_scorces2(), color="deeppink", label="Reconstructed face score", **kwargs)
    plt.legend()
    plt.savefig("lfw_Res50_mapping_type2.svg")

draw_pic1()
draw_pic2()