import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'# 只显示 Error
import logging
logging.disable(30)# for disable the warnning in gradient tape
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
import tensorflow as tf
from tensorflow.keras import Model, optimizers, layers, losses
from ModelZoo import loadFaceModel, loadStyleGAN2Model,createlatent2featureModel,createlatent2featureModelfake, laten2XFinalModel
from PIL import Image
from stylegan2.utils import postprocess_images

num_epochs = 1000
batch_size = 32
# GLOBAL_BATCH_SIZE = batch_size * strategy.num_replicas_in_sync

learning_rate = 0.001

# def createMobdel():
#     inputs = Input((512))# feature
#     initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
#     latents = tf.Variable(initial_value=initializer(shape=[3,512]))
#     image_out = g_clone([latents, []], training=False,
#                         truncation_psi=0.5)
#     image_out = postprocess_images(image_out)
#     dimage_out = tf.image.resize(image_out,
#                                       size=(112, 112))
#     feature_new = arcfacemodel(dimage_out)
#     model = Model(inputs=[], outputs=[feature_new,image_out])
#     return model


# print(inp)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model = laten2XFinalModel()
# model.trainable = True
# model = mytestModel()
print(model.summary())

# 1 st
ranom_fix_z = tf.Variable(np.random.normal(size=(1, 512)), dtype=tf.float32)
print('ranom_fix_z GT',ranom_fix_z)
image_out_gt = model(ranom_fix_z)
gt_img = tf.cast(image_out_gt, dtype=tf.dtypes.uint8)
gt_img = gt_img.numpy()
# image_out = image_out / 255
# print(image_out)
Image.fromarray(gt_img[0], 'RGB').save(
    'data/test6/GT_out.png')

inp = tf.Variable(np.random.normal(size=(1, 512)), dtype=tf.float32)
for i in range(400):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(inp)
        # print(np.linalg.norm(inp))
        clipping_mask = tf.math.logical_or(inp > 1.5, inp < -0.25)
        clipped_values = tf.where(clipping_mask, tf.random.normal(shape=inp.shape), inp)
        inp = tf.compat.v1.assign(inp, clipped_values)
        image_out = model(inp)
        # image_out = g_clone([inp, []], training=False,
        #                     truncation_psi=0.5)
        # img_out = postprocess_images(image_out)
        # dimage_out = tf.image.resize(img_out,
        #                              size=(112, 112))
        # feature_new = arcfacemodel(dimage_out)
        # print(feature_distance)
        image_out_convert = tf.cast(image_out, dtype=tf.dtypes.uint8)
        image_out_convert = image_out_convert.numpy()
        # image_out = image_out / 255
        # print(image_out)
        Image.fromarray(image_out_convert[0], 'RGB').save(
            'data/test6/out_' + str(i) +'_' + '.png')
        # print(feature_new.shape)
        # print(y.shape)
        loss = tf.reduce_mean(tf.losses.mse(image_out, image_out_gt))
        print("epoch %d: loss %f"% (i, loss))
        # print(inp)

    grads = tape.gradient(loss, [inp])    # 使用 model.variables 这一属性直接获得模型中的所有变量
    # print(grads)
    optimizer.apply_gradients(grads_and_vars=zip(grads, [inp]))

# print(inp)
# print(model.variables)

#
# initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
# latents = tf.Variable(initial_value=initializer(shape=[3,512]), trainable=True)
#
# # model = createMobdel()
# # 2ed step, iterative update z to find the best fit
# for epoch in range(num_epochs):
#     # latents
#     with tf.GradientTape() as tape:
#         # seed = np.random.randint()
#         image_out = g_clone([latents, []], training=False,
#                             truncation_psi=0.5)
#         img_out = postprocess_images(image_out)
#         dimage_out = tf.image.resize(img_out,
#                                      size=(112, 112))
#         feature_new = arcfacemodel(dimage_out)
#         image_out = img_out.numpy()
#         Image.fromarray(image_out[0], 'RGB').save(
#             'image_out_' + str(epoch) + '.png')
#         loss1 = tf.reduce_mean(losses.mse(feat_gt,feature_new))
#
#         loss = tf.cast(loss1, dtype=tf.float64)
#         # loss = tf.reduce_mean(loss)
#         # print("epoch %d: loss %f" % (epoch, loss1.numpy()))
#         print("epoch %d: loss %f" % (epoch, loss.numpy()))
#     grads = tape.gradient(loss, [latents])
#     optimizer.apply_gradients(grads_and_vars=zip(grads, [latents]))
#
#     # if epoch % 2000 == 0:
#     #     save_path = os.path.join('./models', f'step{epoch}')
    #     tf.saved_model.save(model, save_path)