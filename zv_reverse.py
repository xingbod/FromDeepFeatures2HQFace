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


os.environ['CUDA_VISIBLE_DEVICES'] = '3'

cfg = load_yaml('./arcface_tf2/configs/arc_res50.yaml')


arcfacemodel = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         training=False)

ckpt_path = tf.train.latest_checkpoint('./arcface_tf2/checkpoints/' + cfg['sub_name'])
print("***********",ckpt_path)
if ckpt_path is not None:
    print("[*] load ckpt from {}".format(ckpt_path))
    arcfacemodel.load_weights(ckpt_path)


strategy = tf.distribute.MirroredStrategy()


class myModel(tf.keras.Model):
    with strategy.scope():
        def __init__(self):
            super().__init__()
            # self.flatten = tf.keras.layers.Flatten(input_shape=(None, 1, 512))
            self.dense1 = tf.keras.layers.Dense(units=512, activation=tf.nn.sigmoid)
            self.dense2 = tf.keras.layers.Dense(units=1024, activation=tf.nn.sigmoid)
            self.dense3 = tf.keras.layers.Dense(units=512)
            self.add = tf.keras.layers.Add()
            self.add2 = tf.keras.layers.Add()

        def call(self, inputs):
            x = self.dense1(inputs)
            # x = self.add([x, inputs])
            x = self.dense2(x)
            # x = self.add2([x, inputs])
            output = self.dense3(x)
            return output






ckpt_dir_base = './official-converted'
ckpt_dir_cuda = os.path.join(ckpt_dir_base, 'cuda')
ckpt_dir_ref = os.path.join(ckpt_dir_base, 'ref')




def z2xTest():
    g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir_cuda, custom_cuda=False)
    # seed = np.random.randint()
    for i in range(50):
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
        Image.fromarray(image_out1[0], 'RGB').save(f'./images/image{i}_out1.png')
        feature = arcfacemodel(dimage_out1)
        new_z = model(feature)
        new_z = new_z + feature
        # print(new_z)
        # new_z = new_z.astype(np.float32)
        labels2 = rnd.randn(1, g_clone.labels_dim)
        labels2 = labels2.astype(np.float32)
        image_out2 = g_clone([new_z, labels2], training=False, truncation_psi=0.5)
        image_out2 = postprocess_images(image_out2)
        # image_out2 = tf.image.resize(image_out2, size=(112, 112))
        image_out2 = image_out2.numpy()
        Image.fromarray(image_out2[0], 'RGB').save(f'./images/image{i}_out2.png')


if __name__ == '__main__':
    model = tf.saved_model.load('./models/step298000')
    z2xTest()