import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Model, optimizers, layers, losses


def zv_data():
    pass


class myModel(tf.keras.Model):
    def __init__(self):
        super(myModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(None, 1, 512))
        self.dense1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=512)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x




