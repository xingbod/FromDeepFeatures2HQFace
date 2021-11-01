import os
import numpy as np
import tensorflow as tf
from ModelZoo import loadArcfaceModel, loadStyleGAN2Model
from stylegan2.utils import postprocess_images
from PIL import Image
import random
from tf_utils import allow_memory_growth

allow_memory_growth()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

array = np.random.randn(1, 512)
inp = tf.constant(array, dtype=tf.float32)

valules = np.linspace(start=-1, stop=1, num=20)

arcfacemodel = loadArcfaceModel()
g_clone = loadStyleGAN2Model()

s_array = list()
for s in range(512):
    s_array.append(s)

index = random.sample(s_array, 256)
os.mkdir('./data/mytest/mytest_random_256/images')
os.mkdir('./data/mytest/mytest_random_256/latentz')
os.mkdir('./data/mytest/mytest_random_256/features')
for i in range(len(valules)):
    for j in range(len(index)):
        array[0][index[j]] = valules[i]
    inp = tf.constant(array, dtype=tf.float32)
    print(inp[0][0:512])
    image_out = g_clone([inp, []], training=False, truncation_psi=0.5)
    image_out = postprocess_images(image_out)
    image_out_g = tf.cast(image_out, dtype=tf.dtypes.uint8)
    image_out_g = image_out_g.numpy()
    image_out = tf.image.resize(image_out, size=(112, 112)) / 255.
    feature_new = arcfacemodel(image_out)

    # image_resize = image_out * 255.
    # image_resize = tf.cast(image_resize, dtype=tf.dtypes.uint8)
    # image_resize = image_resize.numpy()
    # Image.fromarray(image_resize[0], 'RGB').save('./data/mytest/images' + f'/image_resize' + str(i) + '.png')
    Image.fromarray(image_out_g[0], 'RGB').save('./data/mytest/mytest_random_256/images' + f'/image_out' + str(i) + '.png')
    np.savetxt(f'./data/mytest/mytest_random_256/latentz/latent{i}.txt', inp.numpy())
    np.savetxt(f'./data/mytest/mytest_random_256/features/feature{i}.txt', feature_new.numpy())
#
#
# __author__ = 'xingbo, it is based on inital version of sidharthgoyal'
#
# import math
# from random import *
# import random
# import numpy
# import copy
#
# increment = 0.05
# startingPoint = numpy.random.random(512)
# point1 = numpy.random.random(512)
# print('target:', point1)


# point2 = [6,4]
# point3 = [5,2]
# point4 = [2,1]
#
# def distance(coords1, coords2):
#     """ Calculates the euclidean distance between 2 lists of coordinates. """
#     # print('coords1',coords1)
#     # print('coords2',coords2)
#     return numpy.sqrt(numpy.sum((coords1 - coords2) ** 2))
#
#
# def sumOfDistances(x, px):
#     d1 = distance(x, px)
#     return d1
#
#
# def newDistance(d1, point1):
#     d1temp = sumOfDistances(d1, point1)
#     d1 = numpy.append(d1, d1temp)
#     return d1
#
#
# minDistance = sumOfDistances(startingPoint, point1)
# flag = True
#
# threshold = 0.4
# i = 1
# lastFitness = 99
# while lastFitness > threshold:
#     d = []
#     old_point = startingPoint
#     for index in range(512):
#         increment_arr = numpy.zeros(512)
#         increment_arr[index] = increment
#         newpoint = startingPoint + increment_arr
#         d1 = newDistance(newpoint, point1)
#         newpoint = startingPoint - increment_arr
#         d2 = newDistance(newpoint, point1)
#         d.append(d1)
#         d.append(d2)
#
#     # print(i,' ', startingPoint[:4])
#     d = numpy.array(d)
#     minimum = min(d[:, 512])
#     if minimum < minDistance:
#         minindex = numpy.argmin(d[:, 512])
#         startingPoint = d[minindex, :512]
#         minDistance = minimum
#         print('found ', i, ' ', startingPoint[:4], 'score', d[minindex, 512])
#         lastFitness = d[minindex, 512]
#     else:
#         flag = False
#         # print('new start poiny ',i)
#         startingPoint = startingPoint + numpy.random.random(512) * 0.1
#     i += 1
#
# print('target:', point1[:10])
# print('result:', startingPoint[:10])

