import os
import numpy as np
import tensorflow as tf
from ModelZoo import loadArcfaceModel, loadStyleGAN2Model, mytestModel
from stylegan2.utils import postprocess_images

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

with tf.device('/gpu:0'):
    arcfacemodel = loadArcfaceModel()

with tf.device('/gpu:1'):
    g_clone = loadStyleGAN2Model()

with tf.device('/gpu:1'):
    mymodel = mytestModel()


num_pairs = 50000
print('-----------')
latents = tf.constant(np.random.randn(num_pairs, 512), dtype=tf.float32)
images_out = g_clone([latents, []], training=False, truncation_psi=0.5)
images_out = postprocess_images(images_out)
features = arcfacemodel(tf.image.resize(images_out, size=(112, 112)) / 255.)
print('************')





