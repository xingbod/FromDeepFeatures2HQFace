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





arcfacemodel = loadFaceModel()
model = laten2featureFinalModel()
print(model.summary())


def GAalgo(population,crossover_mat_ph,mutation_val_ph):
    # Calculate fitness (MSE)
    # population -> v
    # print("xxxx*******1",population)
    feature_new, image_out = model(population)
    # print("xxxx*******2")
    fitness = -tf.reduce_mean(tf.square(tf.subtract(feature_new, truth_ph)), 1)
    result = tf.math.top_k(fitness, k=pop_size)
    top_vals, top_ind = result.values, result.indices
    # Get best fit individual
    best_val = tf.reduce_min(top_vals)
    best_ind = tf.argmin(top_vals, 0)
    best_individual = tf.gather(population, best_ind)

    best_img = image_out[best_ind]

    # Get parents
    population_sorted = tf.gather(population, top_ind)
    parents = tf.slice(population_sorted, [0, 0], [num_parents, features])

    # Get offspring
    # Indices to shuffle-gather parents
    rand_parent1_ix = np.random.choice(num_parents, num_children)
    rand_parent2_ix = np.random.choice(num_parents, num_children)
    # Gather parents by shuffled indices, expand back out to pop_size too
    rand_parent1 = tf.gather(parents, rand_parent1_ix)
    rand_parent2 = tf.gather(parents, rand_parent2_ix)
    rand_parent1_sel = tf.multiply(rand_parent1, crossover_mat_ph)
    rand_parent2_sel = tf.multiply(rand_parent2, tf.subtract(1., crossover_mat_ph))
    children_after_sel = tf.add(rand_parent1_sel, rand_parent2_sel)

    # Mutate Children
    mutated_children = tf.add(children_after_sel, mutation_val_ph)

    # Combine children and parents into new population
    new_population = tf.concat(axis=0, values=[parents, mutated_children])
    population.assign(new_population)
    return population,best_individual,best_val,fitness,best_img


for i in range(6):
    selection = 0.1 + (i / 10)  # 筛选前20
    for j in range(6):
        # Genetic Algorithm Parameters
        pop_size = 36  # 种群大小
        features = 512  # 个体大小
        selection = 0.1  # 筛选前20
        mutation = (1. + j) / pop_size
        generations = 1000
        num_parents = int(pop_size * selection)
        num_children = pop_size - num_parents


        the_img_savepath = f"./data/auto_test/selection_{i+1}_mutation_{j+1}"
        os.mkdir(the_img_savepath)
        dirs_name = os.listdir(f"./data/stylegan_data_10_exam")  # 人名文件夹列表
        for name in dirs_name:
            dir_path = os.path.join("./data/stylegan_data_10_exam", name)  # 人名目录   img_test{number}

            img_final_path = os.path.join(the_img_savepath, name)
            os.mkdir(img_final_path)

            img_name_list = os.listdir(dir_path)
            img_name = img_name_list[0]
            img_path = os.path.join(dir_path, img_name)
            img = io.imread(img_path)
            img = resize(img, (112, 112), anti_aliasing=True)
            img = np.array(img)
            img = np.expand_dims(img, 0)
            feature_gt = arcfacemodel(img)
            truth = feature_gt
            # Initialize population array
            population = tf.Variable(np.random.randn(pop_size, features), dtype=tf.float32)
            # Initialize placeholders
            truth_ph = truth

            # Run through generations
            pre_fit = 0.0
            new_fit = 0.0
            num = 0
            for i in range(generations):
                # Create cross-over matrices for plugging in.
                crossover_mat = np.ones(shape=[num_children, features])
                crossover_point = np.random.choice(np.arange(1, features - 1, step=1), num_children)
                for pop_ix in range(num_children):
                    crossover_mat[pop_ix, 0:crossover_point[pop_ix]] = 0.
                # Generate mutation probability matrices
                mutation_prob_mat = np.random.uniform(size=[num_children, features])
                mutation_values = np.random.normal(size=[num_children, features])
                mutation_values[mutation_prob_mat >= mutation] = 0

                # Run GA step
                # TF2.0
                population, best_individual, best_val, fitness, best_img = GAalgo(population, crossover_mat,
                                                                                  mutation_values)
                # feed_dict = {truth_ph: truth.reshape([1, features]),
                #              crossover_mat_ph: crossover_mat,
                #              mutation_val_ph: mutation_values}
                image_out = tf.cast(best_img, dtype=tf.dtypes.uint8)
                image_out = image_out.numpy()
                # image_out = image_out / 255
                # print(image_out)
                # if i % 5 == 0:
                best_fit = tf.reduce_min(fitness)

                while (i == 0):
                    pre_fit = -best_fit
                    break

                best_fit_numpy = -(best_fit.numpy())
                print('Generation: {}, Best Fitness (lowest MSE): {:.4}'.format(i, -best_fit))
                if i % 5 == 0:
                    Image.fromarray(image_out, 'RGB').save(
                        img_final_path + r'/out_' + str(i) + "_" + str(format(best_fit_numpy, '.2f')) + '.png')
                new_fit = -best_fit
                if new_fit >= pre_fit:
                    num = num + 1
                else:
                    pre_fit = new_fit
                    num = 0
                if num >= 20:
                    break