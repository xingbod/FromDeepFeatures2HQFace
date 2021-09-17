import os
import numpy as np
import cv2
import pickle
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


# 计算两张图的相似度
def euc_sim(a, b):
    return 1 - np.linalg.norm(a - b)/(np.linalg.norm(a) + np.linalg.norm(b))


