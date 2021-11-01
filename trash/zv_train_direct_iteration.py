import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3, 4, 5'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'# 只显示 Error
import logging
logging.disable(30)# for disable the warnning in gradient tape
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, optimizers, layers, losses
from PIL import Image
from stylegan2.utils import postprocess_images
from ModelZoo import loadFaceModel,loadStyleGAN2Model
# from load_models import load_generator
# from arcface_tf2.modules.models import ArcFaceModel
# from arcface_tf2.modules.utils import set_memory_growth, load_yaml, l2_norm
# from zv_reverse import myModel
# strategy = tf.distribute.MirroredStrategy()

num_epochs = 1000
batch_size = 32
# GLOBAL_BATCH_SIZE = batch_size * strategy.num_replicas_in_sync

learning_rate = 0.00001

arcfacemodel = loadFaceModel()
g_clone = loadStyleGAN2Model()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


# 1st step, extract features from the images based on arcface model
input_dir = "../data/imgs"
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
feat_gt = arcfacemodel(imgs)
batch_size = 3
# 2ed step, iterative update z to find the best fit
for epoch in range(num_epochs):
    # latents
    with tf.GradientTape() as tape:
        # seed = np.random.randint()
        rnd = np.random.RandomState()
        latents = rnd.randn(batch_size, 512)
        latents = tf.convert_to_tensor(latents, dtype=tf.float64)

        image_out = g_clone([latents, []], training=False,
                             truncation_psi=0.5)
        image_out = postprocess_images(image_out)
        dimage_out = tf.image.resize(image_out,
                                      size=(112, 112))
        image_out = image_out.numpy()
        Image.fromarray(image_out[0], 'RGB').save(
            'image_out_'+num_epochs+'.png')
        feature_new = arcfacemodel(dimage_out)


        loss1 = tf.reduce_mean(losses.mse(y_true=feat_gt, y_pred=feature_new))

        loss = tf.cast(loss1, dtype=tf.float64)
        # loss = tf.reduce_mean(loss)
        # print("epoch %d: loss %f" % (epoch, loss1.numpy()))
        print("epoch %d: loss %f" % (epoch, loss.numpy()))
    grads = tape.gradient(loss,latents)
    optimizer.apply_gradients(grads_and_vars=zip(grads, latents))

    # if epoch % 2000 == 0:
    #     save_path = os.path.join('./models', f'step{epoch}')
    #     tf.saved_model.save(model, save_path)

