import random
from tensorflow import keras
import numpy as np
import pickle
import os
# 32 pairs per pickle file

class ZVDatasets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, name_list, batch_size=32):
        self.batch_size = batch_size
        self.batch_size_total = batch_size * 32
        self.name_list = name_list
    def __len__(self):
        return len(self.name_list) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        my_name_list = self.name_list[i : i + self.batch_size]
        latents_data = []
        feature_date = []
        for name in my_name_list:
            with open(os.path.join('./data/rep_try/pairs', f'{name}'), 'rb') as f_read:
                zv_pairs = pickle.load(f_read)
            latents = zv_pairs['z']
            features = zv_pairs['v']
            for i in range(32):
                latents_data.append(latents[i][:])
                feature_date.append(features[i][:])

        X_train, Y_train = np.array(feature_date), np.array(latents_data)

        return  X_train, Y_train
