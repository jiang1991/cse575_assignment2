# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random

import matplotlib.pyplot as plt
# from keras.datasets import mnist
from numpy import *
# import numpy as np
from scipy.io import loadmat
import scipy.stats as st

from k_means import *

data_name = "./AllSamples.mat"

if __name__ == '__main__':
    # 加载数据
    mat_data = loadmat(data_name)
    # print(mat_data.keys())  # 找出keys，读取数据
    data = mat_data['AllSamples']
    print(type(data))

    # plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
    # plt.show()

    centroids = init_centroids(data, 9, 2)
    for k in range(2, 10):
        # print(data)
        # 初始中心
        cluster_assign = k_means(data, k, centroids)

        show_cluster(data, k, centroids, cluster_assign)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
