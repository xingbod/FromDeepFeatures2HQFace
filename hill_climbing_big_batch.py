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
__author__ = 'xingbo, it is based on inital version of sidharthgoyal'
import math
from random import *
import random
import numpy
import os
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tqdm
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'# 只显示 Error
import logging
logging.disable(30)# for disable the warnning in gradient tape
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from ModelZoo import loadStyleGAN2Model, loadArcfaceModel
from stylegan2.utils import postprocess_images
import time
##here we load generator
from tf_utils import allow_memory_growth

# allow_memory_growth()
with tf.device('/gpu:0'):
    arcfacemodel = loadArcfaceModel()

with tf.device('/gpu:1'):
    g_clone = loadStyleGAN2Model()

# Create ground truth
# truth = np.sin(2 * np.pi * (np.arange(features, dtype=np.float32)) / features)
input_dir = "data/imgs"
input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)

imgs =[]
for j, path in enumerate(input_img_paths):
    img = io.imread(path)
    img = resize(img,
           (112,112),
           anti_aliasing=True)
    imgs.append(np.array(img))
imgs = np.array(imgs)
feat_gt_orig = arcfacemodel(imgs/255.0).numpy()
feat_gt = feat_gt_orig[0]

# Hill Climbing Algorithm Parameters
one_batch_size = 32
increment = 0.01
one_batch_size_arc = 512 # 2 *1024
minDistance = 999
threshold = 0.4
the_img_savepath = 'data/hillclimbing'
startingPoint = numpy.random.random(512)
point1 = feat_gt
print('target:', point1[:16])


# point2 = [6,4]
# point3 = [5,2]
# point4 = [2,1]

def distance(coords1, coords2):
    """ Calculates the euclidean distance between 2 lists of coordinates. """
    # print('coords1',coords1)
    # print('coords2',coords2)
    return numpy.sqrt(numpy.sum((coords1 - coords2) ** 2))

def sumOfDistances(x, px):
    d1 = distance(x, px)
    return d1

def newDistance(d1, point1):
    d1temp = sumOfDistances(d1, point1)
    d1 = numpy.append(d1, d1temp)
    return d1

i = 1
lastFitness = 99
while lastFitness > threshold:
    time1 = time.time()
    d = []
    old_point = startingPoint
    new_points = []
    for index in range(512):
        increment_arr = numpy.zeros(512)
        increment_arr[index] = increment
        newpoint = startingPoint + increment_arr
        new_points.append(newpoint)
        newpoint = startingPoint - increment_arr
        new_points.append(newpoint)

    # indeed we have 512*2 new z, need to generate the features
    # one_batch_size = 16
    new_points = np.array(new_points)
    feature_new = np.zeros((1024,512))
    image_out = np.zeros((1024,112,112,3))
    for batch_idx in range(64):
        input = new_points[batch_idx*one_batch_size:(batch_idx+1)*one_batch_size,:]# tf.Variable(np.random.randn(32, features), dtype=tf.float32)
        image_out_g = g_clone([input, []], training=False, truncation_psi=0.5)
        image_out_g = postprocess_images(image_out_g)
        image_out_g = tf.cast(image_out_g, dtype=tf.dtypes.uint8)
        # pay attention to the slice index number
        img_112 = tf.image.resize(image_out_g, size=(112, 112), method='nearest', antialias=True).numpy()
        # print('img_112:',img_112.shape)
        # Image.fromarray(img_112[0], 'RGB').save(
        #     the_img_savepath + r'/image_out_g_' + str(i) + "_" + str(batch_idx) + '.png')
        image_out[batch_idx * one_batch_size:(batch_idx + 1) * one_batch_size, :, :, :] = img_112
    # batch size for arcface 128
    # for batch_idx in range(2):
    #     feature_new[batch_idx * one_batch_size_arc:(batch_idx + 1) * one_batch_size_arc, :] = arcfacemodel(image_out[batch_idx * one_batch_size_arc:(batch_idx + 1) * one_batch_size_arc, :, :, :] ).numpy()
    arc_input = image_out / 255.
    feature_new = arcfacemodel(arc_input).numpy()

    d = np.sqrt(np.sum((feature_new - point1)**2,axis = 1 ))
    # for index in range(512*2):
    #     d2 = newDistance(feature_new[index], point1)
    #     d.append(d2)
    # print(i,' ', startingPoint[:4])
    minimum = min(d)
    if minimum < minDistance:
        minindex = numpy.argmin(d)
        startingPoint = new_points[minindex, :] # latest points
        minDistance = minimum
        print('found ', i, ' ', startingPoint[:4], 'score', d[minindex])
        time2 = time.time()
        time_gap = time2 - time1
        print('Generation: {}, time: {:.2}, Best Fitness (lowest MSE): {:.4}'.format(i, time_gap,d[minindex]))
        lastFitness = d[minindex]
        Image.fromarray(image_out[minindex], 'RGB').save(
            the_img_savepath + r'/out_' + str(i) + "_" + str(format(lastFitness, '.2f')) + '.png')

    else:
        # print('new start poiny ',i)
        startingPoint = startingPoint + numpy.random.random(512) * 0.1
    i += 1

print('target:', point1[:10])
print('result:', startingPoint[:10])

