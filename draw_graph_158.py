import os
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import itertools
# import cv2
import seaborn as sns
from pyeer.eer_info import get_eer_stats

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'# 只显示 Error

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
    features_path = "./data/colorferet_158_features_Xception"
    names_list = os.listdir(features_path)
    imposter_scores = []
    for i,j in nchoosek(0, len(names_list)-1, step=1, n=2):
        with open(os.path.join(features_path, names_list[i], 'source_feature', 'source.pickle'), 'rb') as file:
            features1 = np.array(pickle.load(file))[:2]
        with open(os.path.join(features_path, names_list[j], 'source_feature', 'source.pickle'), 'rb') as file:
            features2 = np.array(pickle.load(file))[:2]
        imposter_scores.append(euc_sim(features1 , features2))
    imp = open(f"./data/scores/colorferet_imposter_score_Xception.pickle", "wb")
    pickle.dump(imposter_scores, imp)
    imp.close()
    # print(imposter_scores[:20])
    return imposter_scores


def cal_genuine_scorces():
    features_path = "./data/colorferet_158_features_Xception"
    names_list = os.listdir(features_path)
    genuine_scores = []
    for name in names_list:
        with open(os.path.join(features_path, name, 'source_feature', 'source.pickle'), 'rb') as file:
            features_list = np.array(pickle.load(file))
        for i, j in nchoosek(0, len(features_list)-1, step=1, n=2):
            feature1 = features_list[i]
            feature2 = features_list[j]
            genuine_scores.append(euc_sim(feature1, feature2))
    gen = open(f"./data/scores/colorferet_genuine_score_Xception.pickle", "wb")
    pickle.dump(genuine_scores, gen)
    gen.close()
    # print(genuine_scores[:20])
    return genuine_scores


def cal_attack_scorces():
    attack_scores = []
    features_path = "./data/celeba_158_features_3_Inception_1"
    names_list = os.listdir(features_path)
    for name in names_list:
        with open(os.path.join(features_path, name, 'gt_feature', 'gt.pickle'), 'rb') as gt_file:
            gt_feature = np.array(pickle.load(gt_file))
        with open(os.path.join(features_path, name, 'pred_feature', 'pred.pickle'), 'rb') as pred_file:
            pred_feature = np.array(pickle.load(pred_file))
        attack_scores.append(euc_sim(gt_feature, pred_feature))

    features_path2 = "./data/celeba_158_features_3_Inception_2"
    names_list2 = os.listdir(features_path2)
    for name2 in names_list2:
        with open(os.path.join(features_path2, name2, 'gt_feature', 'gt.pickle'), 'rb') as gt_file:
            gt_feature = np.array(pickle.load(gt_file))
        with open(os.path.join(features_path2, name2, 'pred_feature', 'pred.pickle'), 'rb') as pred_file:
            pred_feature = np.array(pickle.load(pred_file))
        attack_scores.append(euc_sim(gt_feature, pred_feature))

    features_path3 = "./data/celeba_158_features_3_Inception_3"
    names_list3 = os.listdir(features_path3)
    for name2 in names_list2:
        with open(os.path.join(features_path2, name2, 'gt_feature', 'gt.pickle'), 'rb') as gt_file:
            gt_feature = np.array(pickle.load(gt_file))
        with open(os.path.join(features_path2, name2, 'pred_feature', 'pred.pickle'), 'rb') as pred_file:
            pred_feature = np.array(pickle.load(pred_file))
        attack_scores.append(euc_sim(gt_feature, pred_feature))
    # print(attack_scores[:20])
    att1 = open(f"./data/scores/celeba_attack1_score_Inception.pickle", "wb")
    pickle.dump(attack_scores, att1)
    att1.close()
    return attack_scores


def cal_attack_scorces2():
    features_path = "./data/celeba_158_features_3_Inception_1"
    names_list = os.listdir(features_path)
    attack_scores2 = []
    for name in names_list:
        with open(os.path.join(features_path, name, 'pred_feature', 'pred.pickle'), 'rb') as pred_file:
            pred_feature = np.array(pickle.load(pred_file))
        with open(os.path.join(features_path, name, 'source_feature', 'source.pickle'), 'rb') as file:
            features_list = np.array(pickle.load(file))
        for i in range(len(features_list)):
            attack_scores2.append(euc_sim(pred_feature, features_list[i]))
    att2 = open(f"./data/scores/celeba_attack2_score_Inception.pickle", "wb")
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
    # sns.distplot(cal_attack_scorces(), color="deeppink", label="Reconstructed face score", **kwargs)
    plt.legend()
    plt.savefig("colorferet_Xception_type1.svg")
    stats_a = get_eer_stats(gen, imp)
    # print(stats_a)
    print('stats_a.eer',stats_a.eer)# 10%

def draw_pic2():
    sns.set_style("white")
    kwargs = dict(kde_kws={'linewidth': 0.001})
    plt.figure(figsize=(10, 7), dpi=80)
    sns.distplot(cal_genuine_scorces(), color="dodgerblue", label="Genuine score", **kwargs)
    sns.distplot(cal_imposter_scorces(), color="orange", label="Imposter score", **kwargs)
    # sns.distplot(cal_attack_scorces2(), color="deeppink", label="Reconstructed face score", **kwargs)
    plt.legend()
    plt.savefig("colorferet_Xception_type2.svg")

if __name__ == '__main__':
    draw_pic1()
    draw_pic2()