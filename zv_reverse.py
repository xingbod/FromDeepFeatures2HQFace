import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import Model, optimizers, layers, losses
from PIL import Image
from stylegan2.utils import postprocess_images
from load_models import load_generator
from arcface_tf2.modules.models import ArcFaceModel
from arcface_tf2.modules.utils import set_memory_growth, load_yaml, l2_norm
from tensorflow.python.client import device_lib
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

cfg = load_yaml('./arcface_tf2/configs/arc_res50.yaml')


arcfacemodel = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         training=False)

ckpt_path = tf.train.latest_checkpoint('./arcface_tf2/checkpoints/' + cfg['sub_name'])
print("***********",ckpt_path)
if ckpt_path is not None:
    print("[*] load ckpt from {}".format(ckpt_path))
    arcfacemodel.load_weights(ckpt_path)


class myModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.flatten = tf.keras.layers.Flatten(input_shape=(None, 1, 512))
        self.dense1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(units=512)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        output = self.dense5(x)
        return output


num_epochs = 500
batch_size = 50
learning_rate = 0.001
model = myModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)




ckpt_dir_base = './official-converted'
ckpt_dir_cuda = os.path.join(ckpt_dir_base, 'cuda')
ckpt_dir_ref = os.path.join(ckpt_dir_base, 'ref')


def train(ckpt_dir, use_custom_cuda, out_fn):
    g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir, custom_cuda=use_custom_cuda)
    for epoch in range(num_epochs):
        # seed = np.random.randint()
        rnd = np.random.RandomState()
        latents = rnd.randn(6, g_clone.z_dim)
        labels = rnd.randn(6, g_clone.labels_dim)
        latents = latents.astype(np.float32)
        labels = labels.astype(np.float32)
        image_out = g_clone([latents, labels], training=False, truncation_psi=0.5)
        image_out = postprocess_images(image_out)
        image_out = tf.image.resize(image_out, size=(112, 112))
        image_out = image_out.numpy()
        feature = arcfacemodel(image_out)

        # latents
        with tf.GradientTape() as tape:
            z_pred = model(feature)
            image_out_pred = g_clone([z_pred, labels], training=False, truncation_psi=0.5)
            image_out_pred = postprocess_images(image_out_pred)
            image_out_pred = tf.image.resize(image_out_pred, size=(112, 112))
            image_out_pred = image_out_pred.numpy()
            feature_pred = arcfacemodel(image_out_pred)
            loss1 = losses.mse(y_true=latents, y_pred=z_pred)
            loss1 = tf.reduce_mean(loss1)
            loss2 = losses.mse(image_out, image_out_pred)
            loss2 = tf.reduce_mean(loss2)
            loss3 = losses.mse(feature, feature_pred)
            loss3 = tf.reduce_mean(loss3)
            loss = loss1 + 0.01*loss2 + loss3
            loss = tf.reduce_mean(loss)
            # print("epoch %d: loss %f" % (epoch, loss.numpy()))
            print("epoch %d: loss %f, loss1 %f, loss2 %f, loss3 %f" % (epoch, loss.numpy(), loss1.numpy(), loss2.numpy(), loss3.numpy()))
        grads = tape.gradient(loss, model.variables)

        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


def z2xTest():
    g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir_cuda, custom_cuda=False)
    # seed = np.random.randint()
    rnd = np.random.RandomState()
    latents = rnd.randn(1, g_clone.z_dim)
    labels1 = rnd.randn(1, g_clone.labels_dim)
    latents = latents.astype(np.float32)
    # print(latents)
    labels1 = labels1.astype(np.float32)
    image_out1 = g_clone([latents, labels1], training=False, truncation_psi=0.5)
    image_out1 = postprocess_images(image_out1)
    dimage_out1 = tf.image.resize(image_out1, size=(112, 112))
    image_out1 = image_out1.numpy()
    Image.fromarray(image_out1[0], 'RGB').save('image_out1.png')
    feature = arcfacemodel(dimage_out1)
    new_z = model(feature)
    # print(new_z)
    # new_z = new_z.astype(np.float32)
    labels2 = rnd.randn(1, g_clone.labels_dim)
    labels2 = labels2.astype(np.float32)
    image_out2 = g_clone([new_z, labels2], training=False, truncation_psi=0.5)
    image_out2 = postprocess_images(image_out2)
    # image_out2 = tf.image.resize(image_out2, size=(112, 112))
    image_out2 = image_out2.numpy()
    Image.fromarray(image_out2[0], 'RGB').save('image_out2.png')


if __name__ == '__main__':
    train(ckpt_dir_cuda, use_custom_cuda=False, out_fn='from-cuda-to-ref.png')
    z2xTest()












