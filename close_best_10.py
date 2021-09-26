import os
from skimage import io
import tensorflow as tf
import pickle
import numpy as np
from skimage.transform import resize
import tqdm
from ModelZoo import loadArcfaceModel, loadStyleGAN2Model
from PIL import Image
from stylegan2.utils import postprocess_images
from tf_utils import allow_memory_growth
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

zv_pairs_path = './data/rep_try/pairs'
zv_pairs_names = os.listdir(zv_pairs_path)

gt_img_path = './data/rep_try/imgs'
gt_img_name = os.listdir(gt_img_path)[0]

allow_memory_growth()

arcfacemodel = loadArcfaceModel()
img = io.imread(os.path.join(gt_img_path,gt_img_name))
img = resize(img, (112, 112), anti_aliasing=True)
img = np.array(img) / 255.
img = np.expand_dims(img, 0)
print(img.shape)
gt_feature = arcfacemodel(img)

latents_data = []
feature_date = []

for name in tqdm.tqdm(zv_pairs_names):
    with open(os.path.join('./data/rep_try/pairs', f'{name}'), 'rb') as f_read:
        zv_pairs = pickle.load(f_read)
        latents = zv_pairs['z']
        features = zv_pairs['v']
        latents_data.extend(latents)
        feature_date.extend(features)

feature_date = np.array(feature_date)
latents_data = np.array(latents_data)
scores =  - np.sum((np.array(feature_date) - np.array(gt_feature)) ** 2, axis=1)
print('scores ',scores.shape)
print('scores  mean',np.mean(scores))
print('feature_date ',feature_date.shape)
print('latents_data ',latents_data.shape)
result = tf.math.top_k(scores, k=10)
top_vals, top_ind = result.values, result.indices
print(top_vals[:10], top_ind[:10])



g_clone = loadStyleGAN2Model()
for i in range(10):
    the_save_path = os.path.join(gt_img_path, f'img{i}')
    if not os.path.exists(the_save_path):
        os.mkdir(the_save_path)
    z = latents_data[top_ind[i].numpy(),:]
    np.savetxt(the_save_path + f'/latent_code{i}.txt', z)
    z = np.expand_dims(z, 0)
    image_out = g_clone([z, []], training=False, truncation_psi=0.5)
    image_out = postprocess_images(image_out)
    image_out = tf.cast(image_out, dtype=tf.dtypes.uint8)
    image_out = image_out.numpy()
    Image.fromarray(image_out[0], 'RGB').save(the_save_path + r'/image_out_' + str(i) + '.png')