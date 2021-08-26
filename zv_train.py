import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3, 4, 5'
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
strategy = tf.distribute.MirroredStrategy()

num_epochs = 100000
batch_size = 4
GLOBAL_BATCH_SIZE = batch_size * strategy.num_replicas_in_sync

learning_rate = 0.00001


with strategy.scope():
  # Set reduction to `none` so we can do the reduction afterwards and divide by
  # global batch size.
  loss_object = tf.keras.losses.KLDivergence()
  def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
with strategy.scope():
    model = createModel()
    print(model.summary())
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def train_step(inputs):
    with tf.GradientTape() as tape:
        image_out, feature, new_latent = model(inputs)
        new_latent = tf.cast(new_latent, dtype=tf.float64)
        new_image_out, new_feature, new_new_latent = model(new_latent)
        new_new_latent = tf.cast(new_new_latent, dtype=tf.float64)
        loss1 = compute_loss(inputs, new_latent)
        loss1_2 = compute_loss(new_new_latent, new_latent)
        loss2 = compute_loss(image_out, new_image_out)
        loss3 = compute_loss(feature, new_feature)
        loss = tf.cast(loss1, dtype=tf.float64) + tf.cast(loss1_2, dtype=tf.float64) + 0.01 * tf.cast(loss2,
                                                                                                      dtype=tf.float64) + tf.cast(
            loss3, dtype=tf.float64)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

# `run` replicates the provided computation and runs it
# with the distributed input.
@tf.function
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
  return  tf.reduce_sum(per_replica_losses,) * (1. / GLOBAL_BATCH_SIZE)

EPOCHS = 10000
for epoch in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0
    rnd = np.random.RandomState()
    latents = rnd.randn(GLOBAL_BATCH_SIZE, 512)
    num_batches = num_batches + 1
    total_loss += distributed_train_step(latents)
    train_loss = total_loss / num_batches
    template = ("Epoch {}, Loss: {}")
    print(template.format(epoch + 1, train_loss))

# TRAIN LOOP





#
# for epoch in range(num_epochs):
#     # latents
#     with tf.GradientTape() as tape:
#         # seed = np.random.randint()
#         rnd = np.random.RandomState()
#         latents = rnd.randn(GLOBAL_BATCH_SIZE, 512)
#
#         latents = tf.cast(latents, dtype=tf.float64)
#
#         image_out, feature, new_latent = model(latents)
#
#         new_latent = tf.cast(new_latent, dtype=tf.float64)
#         new_image_out, new_feature, new_new_latent = model(new_latent)
#
#         new_new_latent = tf.cast(new_new_latent, dtype=tf.float64)
#
#         loss1 = tf.reduce_mean(losses.cosine_similarity(y_true=latents, y_pred=new_latent))
#         loss1_2 = tf.reduce_mean( losses.cosine_similarity(y_true=new_new_latent, y_pred=new_latent))
#
#         loss2 = tf.reduce_mean(losses.mae(image_out, new_image_out))
#
#         loss3 = tf.reduce_mean(losses.cosine_similarity(feature, new_feature))
#         # loss3 = tf.reduce_mean(loss3)
#         # print(loss1)
#         # print(loss2)
#         # print(loss3)
#         latents = tf.cast(latents, dtype=tf.float64)
#
#         loss = tf.cast(loss1, dtype=tf.float64)  + tf.cast(loss1_2, dtype=tf.float64)  + 0.01*tf.cast(loss2, dtype=tf.float64) + tf.cast(loss3, dtype=tf.float64)
#         # loss = tf.reduce_mean(loss)
#         # print("epoch %d: loss %f" % (epoch, loss1.numpy()))
#         print("epoch %d: loss %f, loss1 %f,loss1_2 %f, loss2 %f, loss3 %f" % (epoch, loss.numpy(), loss1.numpy(),loss1_2.numpy(), loss2.numpy(), loss3.numpy()))
#     grads = tape.gradient(loss, model.variables)
#     optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
#
#     # if epoch % 2000 == 0:
#     #     save_path = os.path.join('./models', f'step{epoch}')
#     #     tf.saved_model.save(model, save_path)
