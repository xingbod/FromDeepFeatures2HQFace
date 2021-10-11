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
    features_path = "./data/lfw_158_features_Inception"
    names_list = os.listdir(features_path)
    imposter_scores = []
    for i,j in nchoosek(0, len(names_list)-1, step=1, n=2):
        with open(os.path.join(features_path, names_list[i], 'source_feature', 'source.pickle'), 'rb') as file:
            features1 = np.array(pickle.load(file))[:2]
        with open(os.path.join(features_path, names_list[j], 'source_feature', 'source.pickle'), 'rb') as file:
            features2 = np.array(pickle.load(file))[:2]
        imposter_scores.append(euc_sim(features1 , features2))
    # imp = open(f"./data/scores/colorferet_imposter_score_Res50.pickle", "wb")
    # pickle.dump(imposter_scores, imp)
    # imp.close()
    print(imposter_scores[:20])
    return imposter_scores


def cal_genuine_scorces():
    features_path = "./data/lfw_158_features_Inception"
    names_list = os.listdir(features_path)
    genuine_scores = []
    for name in names_list:
        with open(os.path.join(features_path, name, 'source_feature', 'source.pickle'), 'rb') as file:
            features_list = np.array(pickle.load(file))
        for i, j in nchoosek(0, len(features_list)-1, step=1, n=2):
            feature1 = features_list[i]
            feature2 = features_list[j]
            genuine_scores.append(euc_sim(feature1, feature2))
    # gen = open(f"./data/scores/colorferet_genuine_score_Res50.pickle", "wb")
    # pickle.dump(genuine_scores, gen)
    # gen.close()
    print(genuine_scores[:20])
    return genuine_scores

#
# def cal_attack_scorces():
#     attack_scores = []
#     features_path = "./data/colorferet_158_features_3_Res50_1"
#     names_list = os.listdir(features_path)
#     for name in names_list:
#         with open(os.path.join(features_path, name, 'gt_feature', 'gt.pickle'), 'rb') as gt_file:
#             gt_feature = np.array(pickle.load(gt_file))
#         with open(os.path.join(features_path, name, 'pred_feature', 'pred.pickle'), 'rb') as pred_file:
#             pred_feature = np.array(pickle.load(pred_file))
#         attack_scores.append(euc_sim(gt_feature, pred_feature))
#
#     # features_path2 = "./data/lfw_158_features_3_Inception_2"
#     # names_list2 = os.listdir(features_path2)
#     # for name2 in names_list2:
#     #     with open(os.path.join(features_path2, name2, 'gt_feature', 'gt.pickle'), 'rb') as gt_file:
#     #         gt_feature = np.array(pickle.load(gt_file))
#     #     with open(os.path.join(features_path2, name2, 'pred_feature', 'pred.pickle'), 'rb') as pred_file:
#     #         pred_feature = np.array(pickle.load(pred_file))
#     #     attack_scores.append(euc_sim(gt_feature, pred_feature))
#     #
#     # features_path3 = "./data/lfw_158_features_3_Inception_3"
#     # names_list3 = os.listdir(features_path3)
#     # for name3 in names_list3:
#     #     with open(os.path.join(features_path2, name3, 'gt_feature', 'gt.pickle'), 'rb') as gt_file:
#     #         gt_feature = np.array(pickle.load(gt_file))
#     #     with open(os.path.join(features_path2, name3, 'pred_feature', 'pred.pickle'), 'rb') as pred_file:
#     #         pred_feature = np.array(pickle.load(pred_file))
#     #     attack_scores.append(euc_sim(gt_feature, pred_feature))
#     print(attack_scores[:20])
#     # att1 = open(f"./data/scores/colorferet_attack1_score_Res50.pickle", "wb")
#     # pickle.dump(attack_scores, att1)
#     # att1.close()
#     return attack_scores


def cal_attack_scorces_mai():
    attack_scores = []
    features_path = "./data/lfw_mai/features_pred_Inception"
    names_list = os.listdir(features_path)
    for name in names_list:
        img_name_list = os.listdir(f"./data/lfw_mai/features_pred_Inception/{name}")
        for img_name in img_name_list:
            with open(os.path.join(features_path, name, img_name), 'rb') as pred_file:
                pred_feature = np.array(pickle.load(pred_file))
            with open(os.path.join("./data/lfw_mai/features_source_Inception", name, img_name), 'rb') as gt_file:
                gt_feature = np.array(pickle.load(gt_file))
            attack_scores.append(euc_sim(gt_feature, pred_feature))
    att1 = open(f"./data/scores/lfw_mai_Inception_attack1.pickle", "wb")
    pickle.dump(attack_scores, att1)
    att1.close()
    return attack_scores

def cal_attack2_scorces_mai():
    attack_scores2 = []
    features_path = "./data/lfw_mai/features_pred_Inception"
    names_list = os.listdir(features_path)
    for name in names_list:
        img_name_list = os.listdir(f"./data/lfw_mai/features_pred_Inception/{name}")
        for img_name in img_name_list:
            with open(os.path.join(features_path, name, img_name), 'rb') as pred_file:
                pred_feature = np.array(pickle.load(pred_file))
            for source_img_name in img_name_list:
                with open(os.path.join("./data/lfw_mai/features_source_Inception", name, source_img_name), 'rb') as gt_file:
                    gt_feature = np.array(pickle.load(gt_file))
                attack_scores2.append(euc_sim(gt_feature, pred_feature))
    att2 = open(f"./data/scores/lfw_mai_Inception_attack2.pickle", "wb")
    pickle.dump(attack_scores2, att2)
    att2.close()
    return attack_scores2
#
# def cal_attack_scorces2():
#     features_path = "./data/colorferet_158_features_3_Res50_1"
#     names_list = os.listdir(features_path)
#     attack_scores2 = []
#     for name in names_list:
#         with open(os.path.join(features_path, name, 'pred_feature', 'pred.pickle'), 'rb') as pred_file:
#             pred_feature = np.array(pickle.load(pred_file))
#         with open(os.path.join(features_path, name, 'source_feature', 'source.pickle'), 'rb') as file:
#             features_list = np.array(pickle.load(file))
#         for i in range(len(features_list)):
#             attack_scores2.append(euc_sim(pred_feature, features_list[i]))
#             if euc_sim(pred_feature, features_list[i]) == 1:
#                 print(name)
#     # att2 = open(f"./data/scores/colorferet_attack2_score_Res50.pickle", "wb")
#     # pickle.dump(attack_scores2, att2)
#     # att2.close()
#     return attack_scores2


def draw_pic1():
    sns.set_style("white")
    kwargs = dict(kde_kws={'linewidth': 0.001})
    plt.figure(figsize=(10, 7), dpi=80)
    sns.distplot(cal_genuine_scorces(), color="dodgerblue", label="Genuine score", **kwargs)
    sns.distplot(cal_imposter_scorces(), color="orange", label="Imposter score", **kwargs)
    sns.distplot(cal_attack_scorces_mai(), color="deeppink", label="Mated-Attack score", **kwargs)
    sns.set(font_scale=1.6)
    plt.xlabel('Similarity Score', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.legend()
    plt.savefig("mai_Inception_Type1.svg")
    stats_a = get_eer_stats(cal_genuine_scorces(), cal_imposter_scorces())
    # print(stats_a)
    print('stats_a.eer',stats_a.eer)# 10%

def draw_pic2():
    sns.set_style("white")
    kwargs = dict(kde_kws={'linewidth': 0.001})
    plt.figure(figsize=(10, 7), dpi=80)
    sns.distplot(cal_genuine_scorces(), color="dodgerblue", label="Genuine score", **kwargs)
    sns.distplot(cal_imposter_scorces(), color="orange", label="Imposter score", **kwargs)
    sns.distplot(cal_attack2_scorces_mai(), color="deeppink", label="Mated-Attack score", **kwargs)
    sns.set(font_scale=1.6)
    plt.xlabel('Similarity Score', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.legend()
    plt.savefig("mai_Inception_Type2.svg")

if __name__ == '__main__':
    # draw_pic1()
    draw_pic2()