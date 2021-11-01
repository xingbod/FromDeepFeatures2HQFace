import os
import logging
logging.disable(30)# for disable the warnning in gradient tape
from load_models import load_generator
from tf_utils import allow_memory_growth
import tensorflow as tf
from ModelZoo import loadArcfaceModel, mytestModel,mytestModel2
from stylegan2.utils import postprocess_images
from tensorflow.keras import optimizers,losses
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import numpy as np
import tqdm
from trash.ZVDatasets import ZVDatasets
allow_memory_growth()

big_batch_size = 500000
latents_data = []
feature_date = []


def creatZvPairs():
    with tf.device('/cpu:0'):
        arcfacemodel = loadArcfaceModel()

    with tf.device('/gpu:0'):
        ckpt_dir_base = '../official-converted'
        ckpt_dir_cuda = os.path.join(ckpt_dir_base, 'cuda')
        g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir_cuda, custom_cuda=False)

    for batch in tqdm.tqdm(range(big_batch_size)):
        input = np.random.randn(4, 512)
        image_out_g = g_clone([input, []], training=False, truncation_psi=0.5)
        image_out_g = postprocess_images(image_out_g)
        feature_new = arcfacemodel(
            tf.image.resize(image_out_g, size=(112, 112)) / 255.).numpy()
        # images = tf.cast(image_out_g, dtype=tf.dtypes.uint8).numpy()
        #
        with open(os.path.join('./data/rep_try/pairs_4', f'zv_pairs{batch}.pickle'), 'wb') as f_write:
            pickle.dump({'z': input, 'v': feature_new}, f_write)
        # print(f'generate zv_pair: {batch}/200000')


def trainModel():
    name_list = os.listdir('./data/rep_try/pairs')
    model = mytestModel()
    learning_rate = 0.01
    optimizer = optimizers.SGD(learning_rate=learning_rate)
    for batch in tqdm.tqdm(range(21844-1844)):
        with open(os.path.join('./data/rep_try/pairs', f'{name_list[batch]}'), 'rb') as f_read:
            zv_pairs = pickle.load(f_read)
        latents = zv_pairs['z']
        # print(latents)
        features = zv_pairs['v']
        # print(features)
        for i in range(32):
            with tf.GradientTape() as tape:
                latents_pred = model(np.expand_dims(features[i][:], 0))
                loss =tf.reduce_mean(losses.mse(latents[i][:], latents_pred))  +  np.mean(np.sum(latents_pred**2,axis = 1))
                loss1 = tf.reduce_mean(losses.mse(latents[i][:], latents_pred))
                # print(f"Epoch {batch}/36666 batch, {i}/32 step loss = {loss.numpy()}")
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        print(f"Epoch {batch}/21844 batch, loss = {loss.numpy()}, loss1 = {loss1.numpy()}")
        #
        # if batch % 1000 == 0:
        #     val_loss = list()
        #     for batch in tqdm.tqdm(list(range(20000, 21844))):
        #         with open(os.path.join('./data/rep_try/pairs', f'{name_list[batch]}'), 'rb') as f_read:
        #             zv_pairs = pickle.load(f_read)
        #         latents = zv_pairs['z']
        #         features = zv_pairs['v']
        #         for i in range(32):
        #             latents_pred = model(np.expand_dims(features[i][:], 0))
        #             loss1 = tf.reduce_mean(losses.mse(latents[i][:], latents_pred))
        #             loss = tf.reduce_mean(losses.mse(latents[i][:], latents_pred)) + np.mean(
        #                 np.sum(latents_pred ** 2, axis=1))
        #             val_loss.append(loss1)
        #     print(f"val_loss = {np.mean(val_loss)}")
            # save_path = os.path.join('./data/rep_try/models', f'batch{batch}')
            # tf.saved_model.save(model, save_path)

#
def train_model():
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
    model = mytestModel2()
    learning_rate = 0.1
    optimizer = optimizers.SGD(learning_rate=learning_rate)

    input_img_paths = './data/rep_try/pairs'
    namelist = os.listdir(input_img_paths)

    train_list = namelist[:18000]
    val_list = namelist[18001:]

    train_gen = ZVDatasets(train_list,batch_size=64)
    val_gen = ZVDatasets(val_list,batch_size=64)
    print('-----------------------------------------------------------')
    model.compile(loss='mse', optimizer=optimizer,metrics = ['mse'])
    print(model.summary())
    model.fit_generator(train_gen, validation_data=val_gen, epochs=100, callbacks=[early_stopping])



def train_again():
    for batch in tqdm.tqdm(range(36666)):
        # model = tf.saved_model.load('./data/rep_try/batch60000')
        model = mytestModel()
        learning_rate = 0.01
        optimizer = optimizers.SGD(learning_rate=learning_rate)
        with open(os.path.join('./data/rep_try/pairs', f'zv_pairs{batch}.pickle'), 'rb') as f_read:
            zv_pairs = pickle.load(f_read)
        latents = zv_pairs['z']
        features = zv_pairs['v']
        for i in range(32):
            with tf.GradientTape() as tape:
                latents_pred = model(features[i][:])
                loss = losses.mse(latents[i][:], latents_pred) +  np.abs(np.mean(np.sum(latents_pred,axis = 1)))
                # print(f"Epoch_again {batch}/36666 batch, {i}/32 step loss = {loss.numpy()}")
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        # if batch % 1000 == 0:
        #     save_path = os.path.join('./data/rep_try/models', f'batch{batch}')
        #     print(f"Epoch_again {batch}/36666 batch, step loss = {loss.numpy()}")
        #     tf.saved_model.save(model, save_path)



if __name__ == '__main__':
    train_model()