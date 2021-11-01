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
from ModelZoo import loadStyleGAN2Model, loadArcfaceModel
from PIL import Image
from stylegan2.utils import postprocess_images
from load_models import load_generator





arcfacemodel = loadArcfaceModel()
g_clone = loadStyleGAN2Model()
# model = laten2featureFinalModel()
# print(model.summary())


def GAalgo(population,crossover_mat_ph,mutation_val_ph):
    # Calculate fitness (MSE)
    # population -> v
    # print("xxxx*******1",population)
    # feature_new, image_out = model(population)      # 将population传入model得到pop_size张feature和image
    # print("xxxx*******2")

    feature_new = np.zeros((pop_size, 512))
    image_out = np.zeros((pop_size, 1024, 1024, 3))
    for batch in range(big_batch_size):
        input = population[batch * one_batch_size:(batch + 1) * one_batch_size,
                :]  # tf.Variable(np.random.randn(32, features), dtype=tf.float32)
        image_out_g = g_clone([input, []], training=False, truncation_psi=0.5)
        image_out_g = postprocess_images(image_out_g)
        # pay attention to the slice index number
        image_out[batch * one_batch_size:(batch + 1) * one_batch_size, :, :, :] = image_out_g.numpy()
        feature_new[batch * one_batch_size:(batch + 1) * one_batch_size, :] = arcfacemodel(
            tf.image.resize(image_out_g, size=(112, 112)) / 255.)

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


for s in range(4):
    selection = 0.1 + (s / 10)  # 筛选前20
    for m in range(5):
        # Genetic Algorithm Parameters
        big_batch_size = 20
        one_batch_size = 16
        pop_size = one_batch_size * big_batch_size  # 种群大小
        features = 512  # 个体大小
        mutation = 3. / 32
        generations = 1000
        num_parents = int(pop_size * selection)
        num_children = pop_size - num_parents


        the_img_savepath = f"./data/auto_test3/selection_{s+1}_mutation_{m+1}"
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
            img_gt = np.array(img)
            Image.fromarray(img_gt, 'RGB').save(img_final_path + r'/gt_' + name + '.png')
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
                crossover_point2 = np.random.choice(np.arange(1, features - 1, step=1), num_children)
                for pop_ix in range(num_children):
                    # crossover_mat[pop_ix, 0:crossover_point[pop_ix]] = 0.
                    # xingbo added extra crossover point, span between crossover_point  crossover_point2 assign 0
                    crossover_mat[pop_ix, crossover_point[pop_ix]:crossover_point2[pop_ix]] = 0.
                    crossover_mat[pop_ix, crossover_point2[pop_ix]:crossover_point[pop_ix]] = 0.
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

                # 计算图片相似度（余弦距离）
                dot = np.sum(np.multiply(img_gt, image_out), axis=1)
                norm = np.linalg.norm(img_gt, axis=1) * np.linalg.norm(image_out, axis=1)
                dist = dot / norm

                best_fit = tf.reduce_min(fitness)

                while (i == 0):
                    pre_fit = -best_fit
                    break

                best_fit_numpy = -(best_fit.numpy())
                print('Generation: {}, Best Fitness (lowest MSE): {:.4}'.format(i, -best_fit))
                if i % 5 == 0:
                    Image.fromarray(image_out, 'RGB').save(
                        img_final_path + r'/Imageout' + str(i) + "_" + str(dist) + str(format(best_fit_numpy, '.2f')) + '.png')
                new_fit = -best_fit
                if new_fit >= pre_fit:
                    num = num + 1
                else:
                    pre_fit = new_fit
                    num = 0
                if num >= 20:
                    break