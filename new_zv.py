import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'# 只显示 Error
import logging
logging.disable(30)# for disable the warnning in gradient tape
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
import tensorflow as tf
from tensorflow.keras import Model, optimizers, layers, losses
from ModelZoo import loadArcfaceModel, loadStyleGAN2Model
from PIL import Image
from stylegan2.utils import postprocess_images
import time
from shutil import copy
import random
from privacy_enhancing_miu import PrivacyEnhancingMIU

pemiu = PrivacyEnhancingMIU(block_size=32)

num_epochs = 500
batch_size = 32
# GLOBAL_BATCH_SIZE = batch_size * strategy.num_replicas_in_sync


arcfacemodel = loadArcfaceModel()
g_clone = loadStyleGAN2Model()




optimizer = optimizers.SGD(learning_rate=1.2)

pre_loss = 0.0
num = 0
for num_repeat in range(50):

    dir = "./data/lfw_select"
    save_dir = './data/outputs/lfw_results_xb'
    dirs_name = os.listdir("./data/lfw_select")  # 人名文件夹列表


    for name in dirs_name:
        inp = tf.Variable(np.random.randn(1, 512), dtype=tf.float32)
        dir_path = os.path.join(dir, name)  # 人名目录

        if not os.path.exists(save_dir + f"/result_SGD_v{num_repeat}"):
            os.mkdir(save_dir + f"/result_SGD_v{num_repeat}")

        the_img_savepath = save_dir + f"/result_SGD_v{num_repeat}/{name}"
        if not os.path.exists(the_img_savepath):
            os.mkdir(the_img_savepath)

        img_name_list = os.listdir(dir_path)

        for m in img_name_list:
            copy(os.path.join(dir_path, m), the_img_savepath)
        img_name = random.sample(img_name_list, 1)      # 随机选一张当作ground truth
        img_path = os.path.join(dir_path, img_name[0])
        img = io.imread(img_path)
        img_gt = np.array(img)
        Image.fromarray(img_gt, 'RGB').save(the_img_savepath + r'/gt_' + name + '.png')
        img = resize(img, (112, 112), anti_aliasing=True)
        img = np.array(img)
        img = np.expand_dims(img, 0)
        feature_gt = arcfacemodel(img)
        truth_ph = pemiu.shuffle(feature_gt.numpy())
        # Initialize population array
        population = tf.Variable(np.random.randn(1, 512), dtype=tf.float32)

        # Run through generations
        pre_fit = 0.0
        new_fit = 0.0
        num = 0
        for i in range(num_epochs):
            with tf.GradientTape() as tape:
                image_out = g_clone([inp, []], training=False, truncation_psi=0.5)
                image_out = postprocess_images(image_out)
                image_out_g = tf.cast(image_out, dtype=tf.dtypes.uint8)
                image_out_g = image_out_g.numpy()
                image_out = tf.image.resize(image_out, size=(112, 112)) / 255.
                feature_new = arcfacemodel(image_out)
                feature_new = pemiu.shuffle(feature_new.numpy())

                loss1 = tf.reduce_mean(tf.square(tf.subtract(feature_new, feature_gt)), 1)
                loss2 = tf.math.abs(tf.reduce_mean(inp))
                loss = loss1 + loss2
                print("epoch %d: loss1 %f,loss2 %f,loss %f" % (i,loss1,loss2, loss))

                if i % 100 == 0:
                    Image.fromarray(image_out_g[0], 'RGB').save(the_img_savepath + '/out' + str(i) + f'_{loss[0].numpy()}' + '.png')
            grads = tape.gradient(loss, [inp])
            optimizer.apply_gradients(grads_and_vars=zip(grads, [inp]))

            # new_loss = loss
            # if new_loss >= pre_loss:
            #     num = num + 1
            # else:
            #     pre_loss = new_fit
            #     num = 0
            # if num >= 20:
            #     break