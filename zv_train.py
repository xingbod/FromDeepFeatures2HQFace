import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'# 只显示 Error
import logging
logging.disable(30)# for disable the warnning in gradient tape


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, optimizers, layers, losses
from PIL import Image
from stylegan2.utils import postprocess_images
from ModelZoo import createModel
# from load_models import load_generator
# from arcface_tf2.modules.models import ArcFaceModel
# from arcface_tf2.modules.utils import set_memory_growth, load_yaml, l2_norm
# from zv_reverse import myModel

num_epochs = 100000
batch_size = 6
learning_rate = 0.00001

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model = createModel()
print(model.summary())
for epoch in range(num_epochs):
    # latents
    with tf.GradientTape() as tape:
        # seed = np.random.randint()
        rnd = np.random.RandomState()
        latents = rnd.randn(batch_size, 512)
        latents = tf.cast(latents, dtype=tf.float64)

        image_out, feature, new_latent = model(latents)

        new_latent = tf.cast(new_latent, dtype=tf.float64)
        new_image_out, new_feature, new_new_latent = model(new_latent)

        new_new_latent = tf.cast(new_new_latent, dtype=tf.float64)

        loss1 = tf.reduce_mean(losses.cosine_similarity(y_true=latents, y_pred=new_latent))
        loss1_2 = tf.reduce_mean( losses.cosine_similarity(y_true=new_new_latent, y_pred=new_latent))

        loss2 = tf.reduce_mean(losses.mae(image_out, new_image_out))

        loss3 = tf.reduce_mean(losses.cosine_similarity(feature, new_feature))
        # loss3 = tf.reduce_mean(loss3)
        # print(loss1)
        # print(loss2)
        # print(loss3)
        latents = tf.cast(latents, dtype=tf.float64)

        loss = tf.cast(loss1, dtype=tf.float64)  + tf.cast(loss1_2, dtype=tf.float64)  + 0.01*tf.cast(loss2, dtype=tf.float64) + tf.cast(loss3, dtype=tf.float64)
        # loss = tf.reduce_mean(loss)
        # print("epoch %d: loss %f" % (epoch, loss1.numpy()))
        print("epoch %d: loss %f, loss1 %f,loss1_2 %f, loss2 %f, loss3 %f" % (epoch, loss.numpy(), loss1.numpy(),loss1_2.numpy(), loss2.numpy(), loss3.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    # if epoch % 2000 == 0:
    #     save_path = os.path.join('./models', f'step{epoch}')
    #     tf.saved_model.save(model, save_path)
