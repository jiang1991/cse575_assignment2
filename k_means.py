import matplotlib.pyplot as plt
# from keras.datasets import mnist
from numpy import *
# import numpy as np
from scipy.io import loadmat
import scipy.stats as st


# 计算两个点的距离
def cal_dist(d1, d2):
    return sqrt(sum(power(d2 - d1, 2)))


# K 均值算法初始值
# 策略 1：从给定样本中随机选择初始中心。
# 策略 2：随机选择第一个中心；对于第 i 个中心 (i>1)，（在所有可能的样本中）选择一个样本，使得所选样本到之前 (i-1) 所有中心的平均距离最大。
def init_centroids(data, init_k, strategy=1):
    num, dim = data.shape
    centroids = zeros((init_k, dim))
    # todo: 策略2
    if strategy == 1:
        for i in range(init_k):
            index = int(random.uniform(0, num))
            centroids[i, :] = data[index, :]
    else:
        # 第一个点
        # 当前index list
        index_list = []
        last_index = random.randint(0, num)
        centroids[0, :] = data[last_index, :]
        index_list.append(last_index)

        # 最多找到k
        for i in range(1, init_k):
            max_dist = 0
            max_index = 0
            # 找出离last_index最远的点且不在当前列表
            for j in range(num):
                if j in index_list:
                    pass
                dist = cal_dist(data[last_index, :], data[j, :])
                if dist > max_dist:
                    max_dist = dist
                    max_index = j

            last_index = max_index
            index_list.append(max_index)
            centroids[i, :] = data[max_index, :]

    print('初始中心点：(策略{})'.format(strategy))
    print(centroids)
    return centroids


# K 均值算法实现
def k_means(data, init_k, centroids):
    num = size(data, 0)
    centroids = centroids[:init_k]

    # 2x2 matrix, column 0: 所属中心点
    # column 1: error
    cluster_assign = mat(zeros((num, 2)))
    has_changed = True

    # 初始中心
    # centroids = init_centroids(data, init_k)

    while has_changed:
        has_changed = False

        # 遍历sample
        for i in list(range(num)):
            min_dist = 1000
            min_index = 0

            # 找出最近的中心点
            for j in range(init_k):
                dist = cal_dist(centroids[j, :], data[i, :])
                if dist < min_dist:
                    min_dist = dist
                    min_index = j

            # 更新簇
            if cluster_assign[i, 0] != min_index:
                has_changed = True
                cluster_assign[i, :] = min_index, min_dist ** 2

        # 更新中心点
        for j in range(init_k):
            p_in_cluster = data[nonzero(cluster_assign[:, 0].A == j)[0]]
            centroids[j, :] = mean(p_in_cluster, axis=0)

    # print(centroids)
    # 目标函数
    # print(cluster_assign)
    target = sum(cluster_assign, axis=0)
    # print(target.shape)
    print('k = {}, finished. target={}'.format(init_k, target))
    return cluster_assign


# 可视化
def show_cluster(data, init_k, centroids, cluster_assign):
    num, dim = data.shape

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^m', '+c', 'sy', 'dy', '<c', 'pm']
    # 画点
    for i in list(range(num)):
        mark_i = int(cluster_assign[i, 0])
        plt.plot(data[i, 0], data[i, 1], mark[mark_i], markersize=2)

    # 画中心点
    # plt.scatter(data[:, 0], data[:, 1],  s=1, alpha=0.5)
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=10, alpha=1)
    # for i in range(k):
    #     plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)

    for i in range(init_k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=8)
    plt.show()
