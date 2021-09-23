import os
import numpy as np
import tensorflow as tf
from ModelZoo import loadArcfaceModel, loadStyleGAN2Model, mytestModel
from stylegan2.utils import postprocess_images
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3,4,5'

one_batch_size = 32
num_gen_epochs = 1000
regression_batch = 512
num_pairs = num_gen_epochs * one_batch_size
z_0 = np.zeros((num_pairs,512))
v_0 = np.zeros((num_pairs,512))
# step 1, generate z v pairs
with tf.device('/gpu:0'):
    arcfacemodel = loadArcfaceModel()
    g_clone = loadStyleGAN2Model()
    for batch in tqdm.tqdm(range(num_gen_epochs)):
        latents = np.random.randn(one_batch_size, 512)
        z_input = latents.astype(np.float32)
        images_out = g_clone([z_input, []], training=False, truncation_psi=0.5)
        images_out = postprocess_images(images_out)
        features = arcfacemodel(tf.image.resize(images_out, size=(112, 112)) / 255.)
        z_0[batch*one_batch_size:(batch+1)*one_batch_size,:] = latents
        v_0[batch*one_batch_size:(batch+1)*one_batch_size,:] = features.numpy()
# step 2: train model f
with tf.device('/gpu:1'):
    mymodel = mytestModel()
    learning_rate = tf.constant(0.001)
    model_batch_size = 256
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True)
    mymodel.compile(optimizer=optimizer, loss='mse')
    tb_callback = TensorBoard(log_dir='logs/',
                              update_freq=model_batch_size * 5,
                              profile_batch=0)
    tb_callback._total_batches_seen = 1
    tb_callback._samples_seen = model_batch_size
    callbacks = [tb_callback]
    dataset = tf.data.Dataset.from_tensor_slices((v_0,z_0)).batch(regression_batch)

    mymodel.fit(dataset,
              epochs=20,
              steps_per_epoch=int(num_pairs/regression_batch),
              callbacks=callbacks,
              initial_epoch=0)


# step 3: generate further cascade of z v pair
z_0 = np.zeros((num_pairs,512))
v_0 = np.zeros((num_pairs,512))
# step 1, generate z v pairs
with tf.device('/gpu:0'):
    arcfacemodel = loadArcfaceModel()
    g_clone = loadStyleGAN2Model()
    for batch in tqdm.tqdm(range(num_gen_epochs)):
        latents = np.random.randn(one_batch_size, 512)
        z_input = latents.astype(np.float32)
        images_out = g_clone([z_input, []], training=False, truncation_psi=0.5)
        images_out = postprocess_images(images_out)
        features = arcfacemodel(tf.image.resize(images_out, size=(112, 112)) / 255.)
        z_0[batch*one_batch_size:(batch+1)*one_batch_size,:] = latents
        v_0[batch*one_batch_size:(batch+1)*one_batch_size,:] = features.numpy()

z_1 = np.zeros((num_pairs,512))
with tf.device('/gpu:1'):
    for batch in tqdm.tqdm(range(num_gen_epochs)):
        z_1[batch*one_batch_size:(batch+1)*one_batch_size,:] = mymodel(v_0[batch*one_batch_size:(batch+1)*one_batch_size,:]).numpy()

# step 3: generate further cascade of z v pair
v_1 = np.zeros((num_pairs,512))
# step 1, generate z v pairs
with tf.device('/gpu:0'):
    for batch in tqdm.tqdm(range(num_gen_epochs)):
        latents = z_1[batch*one_batch_size:(batch+1)*one_batch_size,:]
        z_input = latents.astype(np.float32)
        images_out = g_clone([z_input, []], training=False, truncation_psi=0.5)
        images_out = postprocess_images(images_out)
        features = arcfacemodel(tf.image.resize(images_out, size=(112, 112)) / 255.)
        v_1[batch*one_batch_size:(batch+1)*one_batch_size,:] = features.numpy()
# step 4 train
with tf.device('/gpu:1'):
    dataset = tf.data.Dataset.from_tensor_slices((v_1,z_0)).batch(regression_batch)
    mymodel.fit(dataset,
              epochs=40,
              steps_per_epoch=int(num_pairs/regression_batch),
              callbacks=callbacks)
