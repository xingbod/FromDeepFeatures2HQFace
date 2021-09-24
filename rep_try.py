import os
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops



os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'# 只显示 Error
import logging
logging.disable(30)# for disable the warnning in gradient tape

from tensorflow.keras import Model, optimizers, layers, losses
from ModelZoo import loadStyleGAN2Model, loadArcfaceModel, mytestModel

from stylegan2.utils import postprocess_images

import tqdm
import pickle


big_batch_size = 50000


arcfacemodel = loadArcfaceModel()
g_clone = loadStyleGAN2Model()


for batch in range(big_batch_size):
    input = np.random.randn(32, 512)
    image_out_g = g_clone([input, []], training=False, truncation_psi=0.5)
    image_out_g = postprocess_images(image_out_g)
    feature_new = arcfacemodel(
        tf.image.resize(image_out_g, size=(112, 112)) / 255.).numpy()
    images = tf.cast(image_out_g, dtype=tf.dtypes.uint8).numpy()

    with open(os.path.join('./data/rep_try/pairs', f'zv_pairs{batch}.pickle'), 'wb') as f_write:
        pickle.dump({'z': input, 'v': feature_new}, f_write)
    print(f'generate zv_pair: {batch}/50000')

model = mytestModel()
learning_rate = 0.0001
optimizer = optimizers.SGD(learning_rate=learning_rate)

for batch in tqdm.tqdm(range(big_batch_size)):
    with open(os.path.join('./data/rep_try/pairs', f'zv_pairs{batch}.pickle'), 'rb') as f_read:
        zv_pairs = pickle.load(f_read)
    latents = zv_pairs['z']
    features = zv_pairs['v']
    for i in range(32):
        with tf.GradientTape() as tape:
            latents_pred = model(features[i][:])
            loss =losses.mse(latents[i][:], latents_pred)
            print(f"Epoch {batch}/50000 batch, {i}/32 step loss = {loss.numpy()}")
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    if batch % 1000 == 0:
        save_path = os.path.join('./data/rep_try/models', f'batch{batch}')
        tf.saved_model.save(model, save_path)




