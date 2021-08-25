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
from zv_reverse import myModel

num_epochs = 100000
batch_size = 50
learning_rate = 0.0001
model = myModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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

ckpt_dir_base = './official-converted'
ckpt_dir_cuda = os.path.join(ckpt_dir_base, 'cuda')
ckpt_dir_ref = os.path.join(ckpt_dir_base, 'ref')

g_clone = load_generator(g_params=None, is_g_clone=True, ckpt_dir=ckpt_dir_cuda, custom_cuda=False)

for epoch in range(num_epochs):
    # seed = np.random.randint()
    rnd = np.random.RandomState()
    latents = rnd.randn(32, g_clone.z_dim)
    labels = rnd.randn(32, g_clone.labels_dim)
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
        # image_out_pred = g_clone([z_pred, labels], training=False, truncation_psi=0.5)
        # image_out_pred = postprocess_images(image_out_pred)
        # image_out_pred = tf.image.resize(image_out_pred, size=(112, 112))
        # image_out_pred = image_out_pred.numpy()
        # feature_pred = arcfacemodel(image_out_pred)
        loss1 = losses.mse(y_true=latents, y_pred=z_pred)
        loss1 = tf.reduce_mean(loss1)
        # loss2 = losses.mse(image_out, image_out_pred)
        # loss2 = tf.reduce_mean(loss2)
        # loss3 = losses.mse(feature, feature_pred)
        # loss3 = tf.reduce_mean(loss3)
        # loss = loss1 + 0.01*loss2 + loss3
        # loss = tf.reduce_mean(loss)
        print("epoch %d: loss %f" % (epoch, loss1.numpy()))
        # print("epoch %d: loss %f, loss1 %f, loss2 %f, loss3 %f" % (epoch, loss.numpy(), loss1.numpy(), loss2.numpy(), loss3.numpy()))
    grads = tape.gradient(loss1, model.variables)

    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    if epoch % 2000 == 0:
        save_path = os.path.join('./models', f'step{epoch}')
        tf.saved_model.save(model, save_path)