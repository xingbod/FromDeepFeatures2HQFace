import os
from skimage import io
import tensorflow as tf
import pickle
import numpy as np
from skimage.transform import resize
from ModelZoo import loadArcfaceModel, loadStyleGAN2Model
from PIL import Image
from stylegan2.utils import postprocess_images

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

zv_pairs_path = './data/rep_try/pairs'
zv_pairs_names = os.listdir(zv_pairs_path)

gt_img_path = './data/rep_try/imgs'
gt_img_name = os.listdir(gt_img_path)[0]

arcfacemodel = loadArcfaceModel()
img = io.imread(os.path.join(gt_img_path,gt_img_name))
print(img)
img = resize(img, (112, 112), anti_aliasing=True)
img = np.array(img) / 255.
img = np.expand_dims(img, 0)
gt_feature = arcfacemodel(img)

latents_data = []
feature_date = []

for name in zv_pairs_names:
    with open(os.path.join('./data/rep_try/pairs', f'{name}'), 'rb') as f_read:
        zv_pairs = pickle.load(f_read)
        latents = zv_pairs['z']
        features = zv_pairs['v']
    for i in range(32):
        latents_data.append(latents[i][:])
        feature_date.append(features[i][:])

loss = -tf.reduce_mean(tf.square(tf.subtract(latents_data, feature_date)), 1)
result = tf.math.top_k(loss, k=10)
top_vals, top_ind = result.values, result.indices

g_clone = loadStyleGAN2Model()
for i in range(10):
    the_save_path = os.path.join(gt_img_path, f'img{i}')
    if not os.path.exists(the_save_path):
        os.mkdir(the_save_path)
    z = latents_data[top_ind[i][:]]
    image_out = g_clone([z, []], training=False, truncation_psi=0.5)
    image_out = postprocess_images(image_out)
    image_out = tf.cast(image_out, dtype=tf.dtypes.uint8)
    image_out = image_out.numpy()
    Image.fromarray(image_out, 'RGB').save(the_save_path + r'/image_out_' + str(i) + '.png')
    np.savetxt(the_save_path + f'/latent_code{i}.txt', z.numpy())