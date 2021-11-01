import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,5'
from tensorflow.keras import Model, optimizers, layers, losses
from stylegan2.utils import postprocess_images

from ModelZoo import loadStyleGAN2Model, loadArcfaceModel
from tf_utils import allow_memory_growth

from trash.LossZoo import perceptual_loss

allow_memory_growth()

num_epochs = 300000
batch_size = 4
learning_rate = 0.0001

with tf.device('/gpu:0'):
    arcfacemodel = loadArcfaceModel()

with tf.device('/gpu:1'):
    g_clone = loadStyleGAN2Model()

# model = myModel()

with tf.device('/gpu:1'):
    # model = myModel()
    model = tf.saved_model.load('./models3/step110000')

#
optimizer = tf.keras.optimizers.Adam(learning_rate)


for epoch in range(num_epochs):
    # seed = np.random.randint()
    rnd = np.random.RandomState()
    latents = rnd.randn(batch_size, g_clone.z_dim)
    latents = latents.astype(np.float32)
    image_out = g_clone([latents, []], training=False, truncation_psi=0.5)
    image_out = postprocess_images(image_out)
    image_out = tf.image.resize(image_out, size=(112, 112))
    image_out = image_out.numpy()
    feature = arcfacemodel(image_out)

    # latents
    with tf.GradientTape() as tape:
        z_pred = model(feature)
        z_pred = z_pred + feature
        image_out_pred = g_clone([z_pred, []], training=False, truncation_psi=0.5)
        image_out_pred = postprocess_images(image_out_pred)
        image_out_pred = tf.image.resize(image_out_pred, size=(112, 112))
        image_out_pred = image_out_pred.numpy()
        feature_pred = arcfacemodel(image_out_pred)
        loss1 = tf.reduce_mean(losses.mse(latents, z_pred))
        # loss2 = tf.reduce_mean(losses.mse(image_out, image_out_pred))
        loss2 = perceptual_loss(image_out, image_out_pred)
        loss3 = tf.reduce_mean(losses.mse(feature, feature_pred))
        loss4 = tf.reduce_mean(losses.mse(latents, z_pred) + 0.001 * tf.norm(latents, ord=1))
        latents = tf.cast(latents, dtype=tf.float64)
        loss = tf.cast(loss1, dtype=tf.float64) + tf.cast(loss2, dtype=tf.float64) + tf.cast(loss3, dtype=tf.float64) + tf.cast(loss4, dtype=tf.float64)
        print("epoch %d: loss %f, loss1 %f, loss2 %f, loss3 %f, loss4 %f" % (
        epoch, loss.numpy(), loss1.numpy(),loss2.numpy(), loss3.numpy(), loss4.numpy()))
        # loss2 = losses.mse(image_out, image_out_pred)
        # loss2 = tf.reduce_mean(loss2)
        # loss3 = losses.mse(feature, feature_pred)
        # loss3 = tf.reduce_mean(loss3)
        # loss = loss1 + 0.01*loss2 + loss3
        # loss = tf.reduce_mean(loss)
        # print("epoch %d: loss %f" % (epoch, loss1.numpy()))
        # print("epoch %d: loss %f, loss1 %f, loss2 %f, loss3 %f" % (epoch, loss.numpy(), loss1.numpy(), loss2.numpy(), loss3.numpy()))
    grads = tape.gradient(loss, model.variables)

    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    if epoch % 2000 == 0:
        save_path = os.path.join('./models4', f'step{epoch}')
        tf.saved_model.save(model, save_path)