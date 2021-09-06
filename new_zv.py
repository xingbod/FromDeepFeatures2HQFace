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
from ModelZoo import loadFaceModel, loadStyleGAN2Model,createlatent2featureModel,createlatent2featureModelfake, laten2featureFinalModel
from PIL import Image
from stylegan2.utils import postprocess_images

num_epochs = 1000
batch_size = 32
# GLOBAL_BATCH_SIZE = batch_size * strategy.num_replicas_in_sync

learning_rate = 0.001

arcfacemodel = loadFaceModel()
# g_clone = loadStyleGAN2Model()
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


# 1st step, extract features from the images based on arcface model
input_dir = "data/imgs"
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
feat_gt_orig = arcfacemodel(imgs)
feat_gt = feat_gt_orig[0]
batch_size = 1

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


y = tf.constant(feat_gt)
inp = tf.Variable(np.random.normal(size=(1, 512)), dtype=tf.float32)
# print(inp)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model = laten2featureFinalModel()
# model.trainable = True
# model = mytestModel()
print(model.summary())
for i in range(400):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(inp)
        # print(np.linalg.norm(inp))
        clipping_mask = tf.math.logical_or(inp > 1, inp < -1)
        clipped_values = tf.where(clipping_mask, tf.random.normal(shape=inp.shape), inp)
        inp = tf.compat.v1.assign(inp, clipped_values)
        feature_new, image_out = model(inp)
        # image_out = g_clone([inp, []], training=False,
        #                     truncation_psi=0.5)
        # img_out = postprocess_images(image_out)
        # dimage_out = tf.image.resize(img_out,
        #                              size=(112, 112))
        # feature_new = arcfacemodel(dimage_out)
        feature_distance = tf.reduce_mean(losses.cosine_similarity(y, feature_new)).numpy()
        # print(feature_distance)
        image_out = tf.cast(image_out, dtype=tf.dtypes.uint8)
        image_out = image_out.numpy()
        # image_out = image_out / 255
        # print(image_out)
        Image.fromarray(image_out[0], 'RGB').save(
            'data/test2/out_' + str(i) +'_' + str(feature_distance) + '.png')
        # print(feature_new.shape)
        # print(y.shape)
        loss = tf.losses.mse(y, feature_new)
        print("epoch %d: loss %f"% (i, loss))
        # print(inp)

    grads = tape.gradient(loss, [inp])    # 使用 model.variables 这一属性直接获得模型中的所有变量
    # print(grads)
    optimizer.apply_gradients(grads_and_vars=zip(grads, [inp]))
    tmp = inp + tf.random.normal(shape=(1, 512), dtype=tf.float32) * 0.01
    inp = tf.compat.v1.assign(inp, tmp)

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