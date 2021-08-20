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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
        self.flatten = tf.keras.layers.Flatten(input_shape=(None, 1, 512))
        self.dense1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=512)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.dense3(x)
        return output


num_epochs = 200
batch_size = 50
learning_rate = 0.001
model = myModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)




ckpt_dir_base = './official-converted'
ckpt_dir_cuda = os.path.join(ckpt_dir_base, 'cuda')
ckpt_dir_ref = os.path.join(ckpt_dir_base, 'ref')


def train(ckpt_dir, use_custom_cuda, out_fn):
    for epoch in range(num_epochs):
        g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir, custom_cuda=use_custom_cuda)
        # seed = np.random.randint()
        rnd = np.random.RandomState()# seed
        latents = rnd.randn(16, g_clone.z_dim)
        labels = rnd.randn(16, g_clone.labels_dim)
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
            loss = losses.mse(y_true=latents, y_pred=z_pred)
            loss = tf.reduce_mean(loss)
            print("epoch %d: loss %f" % (epoch, loss.numpy()))
        grads = tape.gradient(loss, model.variables)

        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))



if __name__ == '__main__':
    train(ckpt_dir_cuda, use_custom_cuda=False, out_fn='from-cuda-to-ref.png')




