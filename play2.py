import pickle
import random
import numpy as np

with open('../30hz/avo_act1_30hz.pkl', 'rb') as f_1:
    data = pickle.load(f_1)
    print(data[0].keys())
    print(data[0]['human_joints_3d'])

x_gaussion_noise = np.random.normal(0, 0.1, size=(10, 21, 1))
y_gaussion_noise = np.random.normal(0, 0.1, size=(10, 21, 1))
score_gaussion_noise = np.zeros((10, 21, 1))
gaussion_noise = np.concatenate((x_gaussion_noise, y_gaussion_noise, score_gaussion_noise), axis=2)
print(gaussion_noise.shape)