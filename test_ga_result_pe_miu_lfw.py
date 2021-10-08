import os
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from shutil import copy
from tf_utils import allow_memory_growth

# allow_memory_growth()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
import time
import random
import tqdm
from privacy_enhancing_miu import PrivacyEnhancingMIU
# PrivacyEnhancing with MIU
pemiu = PrivacyEnhancingMIU(block_size=256)


# Genetic Algorithm Parameters
big_batch_size = 8
one_batch_size = 16
pop_size = one_batch_size * big_batch_size     # 种群大小
features = 512      # 个体大小
selection = 0.2     # 筛选前20
# mutation = 3. / (pop_size / 10)
mutation = 3. / 32
generations = 500
num_parents = int(pop_size * selection)
num_children = pop_size - num_parents


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
    # print(feature_new[0:12])

######################################################################################
    # shuffle each vector depending on the block size
    alt = pemiu.shuffle(feature_new)
    # reconstruct a genuine to reference vector alt[0]
    feature_miu = np.zeros((pop_size, 512))
    for ind in range(pop_size):
        feature_miu[ind] = pemiu.reconstruct(truth_ph, alt[ind])
######################################################################################
    fitness = -tf.reduce_mean(tf.square(tf.subtract(feature_miu, truth_ph)), 1)     # 计算每一行feature和gt的fitness，越大越好
    # print(fitness[0:12])
    result = tf.math.top_k(fitness, k=pop_size)
    top_vals, top_ind = result.values, result.indices       # 拿到最好的pop_size个fitness的值和对应索引
    # print(top_vals[0:12])
    # Get best fit individual| 0909, change from reduace_min to max
    ## {
    best_val = tf.reduce_max(top_vals)      # 选出最好的fitness
    # best_ind = tf.argmax(top_vals, 0)       # 拿到最好的fitness的索引
    # print(top_ind[0])
    best_individual = tf.gather(population, top_ind[0])       # 拿到最好的一个latent code
    best_img = image_out[top_ind[0]]      # 拿到最好的一张图片
    ## }
    # Get parents
    population_sorted = tf.gather(population, top_ind)      # 将population按fitness的大小排序
    features_sorted = tf.gather(feature_new, top_ind)       # 将feature按照fitness的大小排序
    # print(features_sorted[0:12])
    # print(population_sorted[0:12])
    parents = tf.slice(population_sorted, [0, 0], [num_parents, features])      # 拿出前num_parents个population作为parents
    # print(parents)
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
    # print(new_population)
    # print(new_population)
    # population.assign(new_population)
    return new_population,best_individual,best_val,fitness,best_img



for num_repeat in range(5):

    dir = "./data/lfw_select"
    save_dir = './data/lfw_results'
    dirs_name = os.listdir(dir)  # 人名文件夹列表


    for name in dirs_name:
        dir_path = os.path.join(dir, name)  # 人名目录

        if not os.path.exists(save_dir + f"/result{num_repeat+1}"):
            os.mkdir(save_dir + f"/result{num_repeat+1}")

        the_img_savepath = save_dir + f"/result{num_repeat+1}/{name}"
        if not os.path.exists(the_img_savepath):
            os.mkdir(the_img_savepath)

        img_name_list = os.listdir(dir_path)

        for m in img_name_list:
            copy(os.path.join(dir_path, m), the_img_savepath)
        img_name = random.sample(img_name_list, 1)      # 随机选一张当作ground truth
        img_path = os.path.join(dir_path, img_name[0])
        img = io.imread(img_path)
        img_gt = np.array(img)
        Image.fromarray(img_gt, 'RGB').save(the_img_savepath + r'/gt_' + name + '.png')
        img = resize(img, (112, 112), anti_aliasing=True)
        img = np.array(img)
        img = np.expand_dims(img, 0)
        truth_ph = arcfacemodel(img)
        truth_ph = pemiu.shuffle(truth_ph.numpy())
        # Initialize population array
        population = tf.Variable(np.random.randn(pop_size, features), dtype=tf.float32)

        # Run through generations
        pre_fit = 0.0
        new_fit = 0.0
        num = 0
        for i in range(generations):
            # print(population[0:12])
            time1 = time.time()
            # Create cross-over matrices for plugging in.
            crossover_mat = np.ones(shape=[num_children, features])     # 交换矩阵
            crossover_point = np.random.choice(np.arange(1, features - 1, step=1), num_children)  # 选取每行的交换点
            crossover_point2 = np.random.choice(np.arange(1, features - 1, step=1), num_children)  # 选取每行的交换点
            for pop_ix in range(num_children):
                if crossover_point[pop_ix] <= crossover_point2[pop_ix]:
                    crossover_mat[pop_ix, crossover_point[pop_ix]:crossover_point2[pop_ix]] = 0.  # 将每行的交换点前面的元素置为0
                else:
                    crossover_mat[pop_ix, crossover_point2[pop_ix]:crossover_point[pop_ix]] = 0.
                    # Generate mutation probability matrices
            mutation_prob_mat = np.random.uniform(size=[num_children, features])
            mutation_values = np.random.normal(size=[num_children, features])
            mutation_values[mutation_prob_mat >= mutation] = 0  # 突变矩阵

            # Run GA step
            # TF2.0
            population, best_individual, best_val, fitness, best_img = GAalgo(population, crossover_mat, mutation_values)
            # print(best_individual[0:5])
            # feed_dict = {truth_ph: truth.reshape([1, features]),
            #              crossover_mat_ph: crossover_mat,
            #              mutation_val_ph: mutation_values}
            image_out = tf.cast(best_img, dtype=tf.dtypes.uint8)
            image_out = image_out.numpy()
            # image_out = image_out / 255
            # print(image_out)
            # if i % 5 == 0:
            tau = 0.5
            best_fit = tf.reduce_max(fitness)  # fitness是负的，越大越好
            # loss_mean = -tf.reduce_mean(fitness)        # 整个population的平均loss，越小越好
            # loss_history[i+1] = loss_mean
            # print(loss_history[i+1])
            # mutation = (3. / 32 + ((loss_mean - tau) / 32) * 5)
            # mutation = mutation.numpy() + (loss_history[i+1] - loss_history[i]) * 0.1

            while (i == 0):
                pre_fit = -best_fit
                break

            time2 = time.time()
            time_gap = time2 - time1

            best_fit_numpy = -(best_fit.numpy())
            print('Generation: {}, time: {:.2}, mutation rate: {}, Best Fitness (lowest MSE): {:.4}'.format(i, time_gap,
                                                                                                            mutation,
                                                                                                            -best_fit))
            if i % 5 == 0:
                Image.fromarray(image_out, 'RGB').save(
                    the_img_savepath + r'/out_' + str(i) + "_" + str(format(best_fit_numpy, '.2f')) + '.png')
            # new_fit = -best_fit
            # if new_fit >= pre_fit:
            #     num = num + 1
            # else:
            #     pre_fit = new_fit
            #     num = 0
            # if num >= 20:
            #     break