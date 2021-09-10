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
from shutil import copy


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'# 只显示 Error
import logging
logging.disable(30)# for disable the warnning in gradient tape
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow.keras import Model, optimizers, layers, losses
from ModelZoo import loadFaceModel, loadStyleGAN2Model,createlatent2featureModel,createlatent2featureModelfake, laten2featureFinalModel
from PIL import Image


# Genetic Algorithm Parameters
pop_size = 36      # 种群大小
features = 512      # 个体大小
selection = 0.1     # 筛选前20
mutation = 3. / pop_size
generations = 1000
num_parents = int(pop_size * selection)
num_children = pop_size - num_parents

arcfacemodel = loadFaceModel()


model = laten2featureFinalModel()
print(model.summary())


def GAalgo(population,crossover_mat_ph,mutation_val_ph):
    # Calculate fitness (MSE)
    # population -> v
    # print("xxxx*******1",population)
    feature_new, image_out = model(population)      # 将population传入model得到pop_size张feature和image
    # print("xxxx*******2")
    fitness = -tf.reduce_mean(tf.square(tf.subtract(feature_new, truth_ph)), 1)     # 计算每一行feature和gt的fitness，越大越好
    result = tf.math.top_k(fitness, k=pop_size)
    top_vals, top_ind = result.values, result.indices       # 拿到最好的pop_size个fitness的值和对应索引
    # Get best fit individual| 0909, change from reduace_min to max
    ## {
    best_val = tf.reduce_max(top_vals)      # 选出最好的fitness
    best_ind = tf.argmax(top_vals, 0)       # 拿到最好的fitness的索引
    best_individual = tf.gather(population, best_ind)       # 拿到最好的一个latent code

    best_img = image_out[best_ind]      # 拿到最好的一张图片
    ## }
    # Get parents
    population_sorted = tf.gather(population, top_ind)      # 将population按fitness的大小排序
    parents = tf.slice(population_sorted, [0, 0], [num_parents, features])      # 拿出前num_parents个population作为parents

    # Get offspring
    # Indices to shuffle-gather parents
    rand_parent1_ix = np.random.choice(num_parents, num_children)
    rand_parent2_ix = np.random.choice(num_parents, num_children)
    # Gather parents by shuffled indices, expand back out to pop_size too
    rand_parent1 = tf.gather(parents, rand_parent1_ix)      # 从parents中随机抽取（可重复）num_children个出来
    rand_parent2 = tf.gather(parents, rand_parent2_ix)
    # crossover_mat_ph 是num_children*features的01矩阵，每一行的前随机个元素为0，其他为1
    rand_parent1_sel = tf.multiply(rand_parent1, crossover_mat_ph)  # rand_parent1每一行的随机前n个元素变为0
    rand_parent2_sel = tf.multiply(rand_parent2, tf.subtract(1., crossover_mat_ph))   # rand_parent2每一行后features-n个元素变为0
    children_after_sel = tf.add(rand_parent1_sel, rand_parent2_sel)     # 得到交叉互换后的children

    # Mutate Children
    mutated_children = tf.add(children_after_sel, mutation_val_ph)      # 交叉互换后的children加上突变矩阵

    # Combine children and parents into new population
    new_population = tf.concat(axis=0, values=[parents, mutated_children])      # 新的population为新的parents加上新的children
    population.assign(new_population)
    return population,best_individual,best_val,fitness,best_img


dirs_name = os.listdir("./data/auto_updata3")       # 人名文件夹列表
print(dirs_name)


loss_history = np.zeros(generations)
loss_history[0] = 4

for name in dirs_name:
    dir_path = os.path.join("./data/auto_updata3", name)        # 人名目录
    img_name_list = os.listdir(dir_path)
    img_name = img_name_list[0]
    img_path = os.path.join(dir_path, img_name)
    img = io.imread(img_path)
    img = resize(img, (112, 112), anti_aliasing=True)
    img = np.array(img)
    img = np.expand_dims(img, 0)
    feature_gt = arcfacemodel(img)
    # Initialize population array
    population = tf.Variable(np.random.randn(pop_size, features), dtype=tf.float32)
    # Initialize placeholders
    truth_ph = feature_gt

    # Run through generations
    pre_fit = 0.0
    new_fit = 0.0
    num = 0
    for i in range(generations):
        # Create cross-over matrices for plugging in.
        crossover_mat = np.ones(shape=[num_children, features])
        crossover_point = np.random.choice(np.arange(1, features - 1, step=1), num_children)        # 选取每行的交换点
        crossover_point2 = np.random.choice(np.arange(1, features - 1, step=1), num_children)  # 选取每行的交换点
        for pop_ix in range(num_children):
            if crossover_point[pop_ix] <= crossover_point2[pop_ix]:
                crossover_mat[pop_ix, crossover_point[pop_ix]:crossover_point2[pop_ix]] = 0.       # 将每行的交换点前面的元素置为0
            else:
                crossover_mat[pop_ix, crossover_point2[pop_ix]:crossover_point[pop_ix]] = 0.
                # Generate mutation probability matrices
        mutation_prob_mat = np.random.uniform(size=[num_children, features])
        mutation_values = np.random.normal(size=[num_children, features])
        mutation_values[mutation_prob_mat >= mutation] = 0              # 突变矩阵

        # Run GA step
        # TF2.0
        population, best_individual, best_val, fitness, best_img = GAalgo(population, crossover_mat, mutation_values)
        # feed_dict = {truth_ph: truth.reshape([1, features]),
        #              crossover_mat_ph: crossover_mat,
        #              mutation_val_ph: mutation_values}
        image_out = tf.cast(best_img, dtype=tf.dtypes.uint8)
        image_out = image_out.numpy()
        # image_out = image_out / 255
        # print(image_out)
        # if i % 5 == 0:
        tau = 0.1
        best_fit = tf.reduce_max(fitness)       # fitness是负的，越大越好
        loss_mean = -tf.reduce_mean(fitness)        # 整个population的平均loss，越小越好
        loss_history[i+1] = loss_mean
        print(loss_history[i+1])
        mutation = 3. / pop_size + ((loss_mean - tau) / pop_size) * 5
        mutation = mutation.numpy() + (loss_history[i+1] - loss_history[i]) * 0.1

        while (i == 0):
            pre_fit = -best_fit
            break

        best_fit_numpy = -(best_fit.numpy())
        print('Generation: {}, mutation rate: {}, Best Fitness (lowest MSE): {:.4}'.format(i, mutation, -best_fit))
        if i % 5 == 0:
            Image.fromarray(image_out, 'RGB').save(
                dir_path + r'/out_' + str(i) + "_" + str(format(best_fit_numpy, '.2f')) + '.png')
        new_fit = -best_fit
        if new_fit >= pre_fit:
            num = num + 1
        else:
            pre_fit = new_fit
            num = 0
        if num >= 20:
            break








