import os
import pickle

import numpy as np


def get_threshold(FAR_rate):
    with open('./data/scores/colorferet_imposter_score_Inception.pickle', 'rb') as f:
        impost_score = np.array(pickle.load(f))
        impost_score_sorted = np.sort(impost_score, axis=0)
        threshold_inx = int(len(impost_score_sorted) * (1 - FAR_rate))
        threshold = impost_score_sorted[threshold_inx - 1]
        # print('length impostet_score = ', len(impost_score_sorted))
        print(f'when FAR = {FAR_rate}, threshold = {threshold}')
        return threshold


def get_normal_TAR(threshold):
    with open('./data/scores/colorferet_genuine_score_Inception.pickle', 'rb') as f:
        genuine_score = np.array(pickle.load(f))
        num = np.sum(genuine_score > threshold)
        # print('length genuine_score = ', len(genuine_score))
        print('Normal TAR', num / len(genuine_score))


def get_Type1_SAR(threshold):
    with open('./data/scores/colorferet_attack1_score_Inception.pickle', 'rb') as f:
        attack1_score = np.array(pickle.load(f))
        num = np.sum(attack1_score > threshold)
        # print('length genuine_score = ', len(genuine_score))
        print('Type1 SAR = ', num / len(attack1_score))


def get_Type2_SAR(threshold):
    with open('./data/scores/colorferet_attack2_score_Inception.pickle', 'rb') as f:
        attack1_score = np.array(pickle.load(f))
        num = np.sum(attack1_score > threshold)
        # print('length genuine_score = ', len(genuine_score))
        print('Type2 SAR = ', num / len(attack1_score))

if __name__ == '__main__':
    threshold = get_threshold(0.1)
    get_normal_TAR(threshold)
    get_Type1_SAR(threshold)
    get_Type2_SAR(threshold)