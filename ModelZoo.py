import tensorflow as tf
import os

from stylegan2.utils import postprocess_images
from load_models import load_generator
from arcface_tf2.modules.models import ArcFaceModel
from arcface_tf2.modules.utils import set_memory_growth, load_yaml, l2_norm

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, BatchNormalization, LeakyReLU, concatenate


def regressionModel():
    inputs = Input((512))
    x = tf.keras.layers.Dense(units=512, activation='sigmoid')(inputs)
    # x = tf.keras.layers.Dense(units=512, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(units=512)(x)
    model = Model(inputs=[inputs], outputs=[x])
    return model




def createModel():
    cfg = load_yaml('./arcface_tf2/configs/arc_res50.yaml')
    arcfacemodel = ArcFaceModel(size=cfg['input_size'],
                                backbone_type=cfg['backbone_type'],
                                training=False)

    ckpt_path = tf.train.latest_checkpoint('./arcface_tf2/checkpoints/' + cfg['sub_name'])
    print("***********", ckpt_path)
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        arcfacemodel.load_weights(ckpt_path)

    arcfacemodel.trainable = False

    ckpt_dir_base = './official-converted'
    ckpt_dir_cuda = os.path.join(ckpt_dir_base, 'cuda')

    g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir_cuda, custom_cuda=False)
    g_clone.trainable = False

    inputs_latents = Input((g_clone.z_dim))

    image_out = g_clone([inputs_latents, []], training=False, truncation_psi=0.5)

    image_out = postprocess_images(image_out)
    image_out = tf.image.resize(image_out, size=(112, 112))

    feature = arcfacemodel(image_out)

    reg_model = regressionModel()

    new_latent = reg_model(feature)

    # 0827 xingbo add residual as the obj
    new_latent = new_latent + inputs_latents

    model = Model(inputs=[inputs_latents], outputs=[image_out, feature, new_latent])
    return model


def loadFaceModel():
    cfg = load_yaml('./arcface_tf2/configs/arc_res50.yaml')
    arcfacemodel = ArcFaceModel(size=cfg['input_size'],
                                backbone_type=cfg['backbone_type'],
                                training=False)

    ckpt_path = tf.train.latest_checkpoint('./arcface_tf2/checkpoints/' + cfg['sub_name'])
    print("***********", ckpt_path)
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        arcfacemodel.load_weights(ckpt_path)

    arcfacemodel.trainable = False
    return arcfacemodel


def loadStyleGAN2Model():
    ckpt_dir_base = './official-converted'
    ckpt_dir_cuda = os.path.join(ckpt_dir_base, 'cuda')

    g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir_cuda, custom_cuda=False)
    g_clone.trainable = False
    return g_clone


# model = createModel()
# print(model.summary())