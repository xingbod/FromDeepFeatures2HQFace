import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'# 只显示 Error
import logging
logging.disable(30)# for disable the warnning in gradient tape
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, BatchNormalization, LeakyReLU, concatenate

from tensorflow.keras import Model, optimizers, layers, losses
from PIL import Image
from stylegan2.utils import postprocess_images
from ModelZoo import loadFaceModel, loadStyleGAN2Model,createlatent2featureModel, mytestModel, createTestModel
from scipy.optimize import minimize


num_epochs = 1000
batch_size = 32
# GLOBAL_BATCH_SIZE = batch_size * strategy.num_replicas_in_sync

learning_rate = 0.00001

arcfacemodel = loadFaceModel()
# g_clone = loadStyleGAN2Model()
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


# 1st step, extract features from the images based on arcface model
input_dir = "data/imgs"
input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)

imgs =[]
for j, path in enumerate(input_img_paths):
    img = io.imread(path)
    img = resize(img,
           (112,112),
           anti_aliasing=True)
    imgs.append(np.array(img))
imgs = np.array(imgs)
feat_gt_orig = arcfacemodel(imgs)
feat_gt = feat_gt_orig[0]
batch_size = 1

y = tf.constant(feat_gt)
inp = tf.Variable(np.random.normal(size=(1, 512)), dtype=tf.float32)
# print(inp)
model = createlatent2featureModel()
# model.trainable = True
# model = mytestModel()
print(model.summary())

def minObj(inp):
    feature_new, imgs = model(inp)
    loss = tf.losses.mse(y, feature_new)
	return loss

res = minimize(minObj, inp, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True})


print(res.x)
