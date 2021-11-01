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


os.environ['CUDA_VISIBLE_DEVICES'] = '4'

cfg = load_yaml('../arcface_tf2/configs/arc_res50.yaml')


arcfacemodel = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         training=False)

ckpt_path = tf.train.latest_checkpoint('./arcface_tf2/checkpoints/' + cfg['sub_name'])
print("***********",ckpt_path)
if ckpt_path is not None:
    print("[*] load ckpt from {}".format(ckpt_path))
    arcfacemodel.load_weights(ckpt_path)


ckpt_dir_base = '../official-converted'
ckpt_dir_cuda = os.path.join(ckpt_dir_base, 'cuda')

model = tf.saved_model.load('./models/step298000')
mypic_path = r'./mypic/test.png'
def pic2pic():
    g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir_cuda, custom_cuda=False)
    img1 = Image.open(mypic_path)
    tf_image = np.array(img1)
    tf_image = tf.expand_dims(tf_image, axis=0)
    tf_image = tf.image.resize(tf_image, size=(112, 112))
    feature = arcfacemodel(tf_image)
    z_pred = model(feature)
    rnd = np.random.RandomState()
    labels = rnd.randn(1, g_clone.labels_dim)
    labels = labels.astype(np.float32)
    image_out = g_clone([z_pred, labels], training=False, truncation_psi=0.5)
    image_out = postprocess_images(image_out)
    # image_out2 = tf.image.resize(image_out2, size=(112, 112))
    image_out = image_out.numpy()
    Image.fromarray(image_out[0], 'RGB').save(f'./mypic/image_out.png')


if __name__ == '__main__':
    pic2pic()