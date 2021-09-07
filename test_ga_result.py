# -*- coding: utf-8 -*-
# Implementing a Genetic Algorithm
# -------------------------------
#
# Genetic Algorithm Optimization in TensorFlow
#
# We are going to implement a genetic algorithm
#   to optimize to a ground truth array.  The ground
#   truth will be an array of 50 floating point
#   numbers which are generated by:
#   f(x)=sin(2*pi*x/50) where 0<x<50
#
# Each individual will be an array of 50 floating
#   point numbers and the fitness will be the average
#   mean squared error from the ground truth.
#
# We will use TensorFlow's update function to run the
#   different parts of the genetic algorithm.
#
# While TensorFlow isn't really the best for GA's,
#   this example shows that we can implement different
#   procedural algorithms with TensorFlow operations.

import os
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'# 只显示 Error
import logging
logging.disable(30)# for disable the warnning in gradient tape
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow.keras import Model, optimizers, layers, losses
from ModelZoo import loadFaceModel, loadStyleGAN2Model,createlatent2featureModel,createlatent2featureModelfake, laten2featureFinalModel
from PIL import Image


# Genetic Algorithm Parameters
pop_size = 32      # 种群大小
features = 512      # 个体大小
selection = 0.2     # 筛选前20
mutation = 1. / pop_size
generations = 2000
num_parents = int(pop_size * selection)
num_children = pop_size - num_parents


# Create ground truth
# truth = np.sin(2 * np.pi * (np.arange(features, dtype=np.float32)) / features)

file_list = os.listdir("D:\data\lfw_mtcnnpy_160\lfw_mtcnnpy_160")
dirs_name = file_list
print(dirs_name)



input_dir = "D:\data/face_img_50"
input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)

# imgs =[]
# for j, path in enumerate(input_img_paths):
#     img = io.imread(path)
#     img = resize(img,
#            (112,112),
#            anti_aliasing=True)
#     imgs.append(np.array(img))
# imgs = np.array(imgs)
# arcfacemodel = loadFaceModel()
# feat_gt_orig = arcfacemodel(imgs)
# feat_gt = feat_gt_orig[0]
# truth = feat_gt
#
#
# # Initialize population array
# population = tf.Variable(np.random.randn(pop_size, features), dtype=tf.float32)
# # Initialize placeholders
# truth_ph = truth
# model = laten2featureFinalModel()
# print(model.summary())
#
# def GAalgo(population,crossover_mat_ph,mutation_val_ph):
#     # Calculate fitness (MSE)
#     # population -> v
#     # print("xxxx*******1",population)
#     feature_new, image_out = model(population)
#     # print("xxxx*******2")
#     fitness = -tf.reduce_mean(tf.square(tf.subtract(feature_new, truth_ph)), 1)
#     result = tf.math.top_k(fitness, k=pop_size)
#     top_vals, top_ind = result.values, result.indices
#     # Get best fit individual
#     best_val = tf.reduce_min(top_vals)
#     best_ind = tf.argmin(top_vals, 0)
#     best_individual = tf.gather(population, best_ind)
#
#     best_img = image_out[best_ind]
#
#     # Get parents
#     population_sorted = tf.gather(population, top_ind)
#     parents = tf.slice(population_sorted, [0, 0], [num_parents, features])
#
#     # Get offspring
#     # Indices to shuffle-gather parents
#     rand_parent1_ix = np.random.choice(num_parents, num_children)
#     rand_parent2_ix = np.random.choice(num_parents, num_children)
#     # Gather parents by shuffled indices, expand back out to pop_size too
#     rand_parent1 = tf.gather(parents, rand_parent1_ix)
#     rand_parent2 = tf.gather(parents, rand_parent2_ix)
#     rand_parent1_sel = tf.multiply(rand_parent1, crossover_mat_ph)
#     rand_parent2_sel = tf.multiply(rand_parent2, tf.subtract(1., crossover_mat_ph))
#     children_after_sel = tf.add(rand_parent1_sel, rand_parent2_sel)
#
#     # Mutate Children
#     mutated_children = tf.add(children_after_sel, mutation_val_ph)
#
#     # Combine children and parents into new population
#     new_population = tf.concat(axis=0, values=[parents, mutated_children])
#     population.assign(new_population)
#     return population,best_individual,best_val,fitness,best_img
#
#
# # Run through generations
# for i in range(generations):
#     # Create cross-over matrices for plugging in.
#     crossover_mat = np.ones(shape=[num_children, features])
#     crossover_point = np.random.choice(np.arange(1, features - 1, step=1), num_children)
#     for pop_ix in range(num_children):
#         crossover_mat[pop_ix, 0:crossover_point[pop_ix]] = 0.
#     # Generate mutation probability matrices
#     mutation_prob_mat = np.random.uniform(size=[num_children, features])
#     mutation_values = np.random.normal(size=[num_children, features])
#     mutation_values[mutation_prob_mat >= mutation] = 0
#
#     # Run GA step
#     # TF2.0
#     population,best_individual,best_val,fitness,best_img = GAalgo(population,crossover_mat,mutation_values)
#     # feed_dict = {truth_ph: truth.reshape([1, features]),
#     #              crossover_mat_ph: crossover_mat,
#     #              mutation_val_ph: mutation_values}
#     image_out = tf.cast(best_img, dtype=tf.dtypes.uint8)
#     image_out = image_out.numpy()
#     # image_out = image_out / 255
#     # print(image_out)
#     # if i % 5 == 0:
#     Image.fromarray(image_out, 'RGB').save(
#             'data/test13/out_' + str(i) + '.png')
#     if i % 5 == 0:
#         best_fit = tf.reduce_min(fitness)
#         print('Generation: {}, Best Fitness (lowest MSE): {:.2}'.format(i, -best_fit))
#
# # plt.plot(truth, label="True Values")
# # plt.plot(np.squeeze(best_individual_val), label="Best Individual")
# # plt.axis((0, features, -1.25, 1.25))
# # plt.legend(loc='upper right')
# # plt.show()