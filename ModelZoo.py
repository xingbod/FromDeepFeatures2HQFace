import tensorflow as tf
import os

from stylegan2.utils import postprocess_images
from load_models import load_generator
from arcface_tf2.modules.models import ArcFaceModel
from arcface_tf2.modules.utils import set_memory_growth, load_yaml, l2_norm

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, BatchNormalization, LeakyReLU, concatenate,Dense,Flatten
from bicubic_downsample import build_filter, apply_bicubic_downsample

def regressionModel():
    inputs = Input((512))
    x = tf.keras.layers.Dense(units=512, activation='sigmoid')(inputs)
    # x = tf.keras.layers.Dense(units=512, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(units=512)(x)
    model = Model(inputs=[inputs], outputs=[x])
    return model

# fake arcfacemodel
def mytestModel():
    inputs = Input((512))
    x = tf.keras.layers.Dense(units=512, activation='relu')(inputs)
    output = tf.keras.layers.Dense(units=512)(x)
    model = Model(inputs=[inputs], outputs=[output])
    model.trainable = False
    return model


def myFakeStyleGan():
    inputs = Input((112, 112))
    x = tf.keras.layers.Dense(units=512, activation='relu')(inputs)
    output = tf.keras.layers.Dense(units=3)(x)
    model = Model(inputs=[inputs], outputs=[output])
    model.trainable = False
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
    model.trainable = True
    return model


# fake styleGan and arcface
def createTestModel():
    cfg = load_yaml('./arcface_tf2/configs/arc_res50.yaml')
    arcfacemodel = ArcFaceModel(size=cfg['input_size'],
                                backbone_type=cfg['backbone_type'],
                                training=False)

    ckpt_path = tf.train.latest_checkpoint('./arcface_tf2/checkpoints/' + cfg['sub_name'])
    print("***********", ckpt_path)
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        arcfacemodel.load_weights(ckpt_path)

    # ckpt_dir_base = './official-converted'
    # ckpt_dir_cuda = os.path.join(ckpt_dir_base, 'cuda')
    #
    # g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir_cuda, custom_cuda=False)
    # g_clone.trainable = True
    inputs_latents = Input((112, 112))

    # image_out = g_clone([inputs_latents, []], training=False, truncation_psi=0.5)

    # image_out = postprocess_images(image_out)
    # image_out = tf.image.resize(image_out, size=(112, 112))
    myfakestyle = myFakeStyleGan()
    image_out = myfakestyle(inputs_latents)
    # mymodel = mytestModel()
    feature = arcfacemodel(image_out)
    model = Model(inputs=[inputs_latents], outputs=[feature])
    return model


def createlatent2featureModel():
    cfg = load_yaml('./arcface_tf2/configs/arc_res50.yaml')
    arcfacemodel = ArcFaceModel(size=cfg['input_size'],
                                backbone_type=cfg['backbone_type'],
                                training=False)

    ckpt_path = tf.train.latest_checkpoint('./arcface_tf2/checkpoints/' + cfg['sub_name'])
    print("***********", ckpt_path)
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        arcfacemodel.load_weights(ckpt_path)

    arcfacemodel.trainable = True

    ckpt_dir_base = './official-converted'
    ckpt_dir_cuda = os.path.join(ckpt_dir_base, 'cuda')

    g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir_cuda, custom_cuda=False)
    g_clone.trainable = True

    inputs_latents = Input((g_clone.z_dim))

    image_out = g_clone([inputs_latents, []], training=False, truncation_psi=0.5)

    image_out = postprocess_images(image_out)

    image_out = tf.cast(image_out, dtype=tf.float32)
    # First, create the bicubic kernel. This can be reused in multiple downsample operations
    k = build_filter(factor=9)

    # Downsample x which is a tensor with shape [N, H, W, 3]
    y = apply_bicubic_downsample(image_out, filter=k, factor=9)
    # print(y[:, :-1, :-1, :])
    # y now contains x downsampled to [N, H/4, W/4, 3]
    # image_out = tf.image.resize(image_out, size=(112, 112))
    feature = arcfacemodel(y[:, :-1, :-1, :])

    model = Model(inputs=[inputs_latents], outputs=[feature, image_out])
    return model



def createlatent2featureModelfake():
    cfg = load_yaml('./arcface_tf2/configs/arc_res50.yaml')
    arcfacemodel = ArcFaceModel(size=cfg['input_size'],
                                backbone_type=cfg['backbone_type'],
                                training=False)

    ckpt_path = tf.train.latest_checkpoint('./arcface_tf2/checkpoints/' + cfg['sub_name'])
    print("***********", ckpt_path)
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        arcfacemodel.load_weights(ckpt_path)

    arcfacemodel.trainable = True


    ckpt_dir_base = './official-converted'
    ckpt_dir_cuda = os.path.join(ckpt_dir_base, 'cuda')

    g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir_cuda, custom_cuda=False)
    g_clone.trainable = True

    inputs_latents = Input((g_clone.z_dim))

    image_out = g_clone([inputs_latents, []], training=False, truncation_psi=0.5)
    image_out = postprocess_images(image_out)

    # inputs_latents = Input((512))
    #
    # x = Dense(512)(inputs_latents)
    # x = Dense(1024*1024*3)(x)
    image_out = tf.keras.layers.Reshape((1024, 1024,3))(image_out)


    # image_out = tf.cast(image_out, dtype=tf.float32)
    # # First, create the bicubic kernel. This can be reused in multiple downsample operations
    # k = build_filter(factor=9)
    #
    # # Downsample x which is a tensor with shape [N, H, W, 3]
    # y = apply_bicubic_downsample(image_out, filter=k, factor=9)
    # print(y[:, :-1, :-1, :])
    # y now contains x downsampled to [N, H/4, W/4, 3]
    image_out = tf.image.resize(image_out, size=(112, 112))
    feature = arcfacemodel(image_out)

    model = Model(inputs=[inputs_latents], outputs=[feature, image_out])
    return model

def loadFaceModel():
    cfg = load_yaml('./arcface_tf2/configs/arc_pre_trained_res100.yaml')
    arcfacemodel = ArcFaceModel(size=cfg['input_size'],
                                backbone_type=cfg['backbone_type'],
                                training=False)

    ckpt_path = tf.train.latest_checkpoint('./arcface_tf2/checkpoints/' + cfg['sub_name'])
    print("****ckpt_path*******", ckpt_path)
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        arcfacemodel.load_weights(ckpt_path)

    arcfacemodel.trainable = True
    return arcfacemodel


def loadArcfaceModel():
    cfg = load_yaml('./arcface_tf2/configs/arc_res50.yaml')
    arcfacemodel = ArcFaceModel(size=cfg['input_size'],
                                backbone_type=cfg['backbone_type'],
                                training=False)

    ckpt_path = tf.train.latest_checkpoint('./arcface_tf2/checkpoints/' + cfg['sub_name'])
    print("***********", ckpt_path)
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        arcfacemodel.load_weights(ckpt_path)

    arcfacemodel.trainable = True
    return arcfacemodel


def loadStyleGAN2Model():
    ckpt_dir_base = './official-converted'
    ckpt_dir_cuda = os.path.join(ckpt_dir_base, 'cuda')

    g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir_cuda, custom_cuda=False)
    g_clone.trainable = True
    return g_clone


def laten2featureFinalModel():
    arcfacemodel = loadFaceModel()
    g_clone = loadStyleGAN2Model()
    inputs_latents = Input((g_clone.z_dim))
    image_out = g_clone([inputs_latents, []], training=False, truncation_psi=0.5)
    image_out_post = postprocess_images(image_out)
    # image_out_1 = tf.image.random_flip_left_right(image_out_post)
    # image_out_1 = tf.image.random_brightness(image_out_1,50)
    # image_out_1 = tf.image.random_contrast(image_out_1,0.2,1.5)
    image_out = tf.image.resize(image_out_post, size=(112, 112))
    feature = arcfacemodel(image_out/255.)
    model = Model(inputs=[inputs_latents], outputs=[feature, image_out_post])
    return model


def laten2XFinalModel():
    g_clone = loadStyleGAN2Model()
    inputs_latents = Input((g_clone.z_dim))
    image_out = g_clone([inputs_latents, []], training=False, truncation_psi=0.5)
    image_out_post = postprocess_images(image_out)

    model = Model(inputs=[inputs_latents], outputs=[image_out_post])
    return model



# model = createModel()
# print(model.summary())