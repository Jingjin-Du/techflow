from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
import copy

def createData():
    X, y = datasets.make_blobs(n_samples=10000, n_features=2, centers=[[-1, -1], [1, 1], [2, 2]], cluster_std=[0.4, 0.5, 0.6])
    return X, y


def calculateDistance(vecA, vecB):
    return np.sum(np.square(vecA - vecB))


def getCloestDist(point, dataset):
    mindist = math.inf
    for data in dataset:
        dist = calculateDistance(point, data)
        if dist < mindist:
            mindist = dist
    return mindist


def randomCenter(samples, K):
    m, n = np.shape(samples)
    centers = np.mat(np.zeros((K, n)))
    for i in range(n):
        mxi = np.max(samples[:, i])
        mni = np.min(samples[:, i])
        rangeI = mxi - mni
        centers[:, i] = np.mat(mni + rangeI * np.random.rand(K, 1))
    return centers


def randomCenterPlus(dataset, k):
    centers = []
    n = dataset.shape[0]
    rdx = np.random.choice(range(n), 1)
    centers.append(np.squeeze(dataset[rdx]).tolist())
    d = [0 for _ in range(len(dataset))]
    for _ in range(1, k):
        tot = 0
        for i, point in enumerate(dataset):
            d[i] = getCloestDist(point, centers)
            tot += d[i]
        
        tot *= random.random()
        for i, di in enumerate(d):
            tot -= di
            if tot > 0:
                continue
            centers.append(np.squeeze(dataset[i]).tolist())
            break

    return np.mat(centers)


def kmeans(dataset, k, plus):
    m, n = np.shape(dataset)
    clusterPos = np.zeros((m, 2))

    centers = []
    if not plus:
        #随机生成初始中心点
        centers = randomCenter(dataset, k)
    else:
        #kmeans++生成初始中心点
        centers = randomCenterPlus(dataset, k)
    centers_bak = copy.deepcopy(centers)
    clusterChange = True
    plt.ion()
    iter = 1
    while clusterChange:
        plt.clf()
        clusterChange = False
        for i in range(m):
            minD = float('inf')
            idx = -1
            for j in range(k):
                dis = calculateDistance(centers[j, :], dataset[i, :])
                if dis < minD:
                    minD = dis
                    idx = j
            if clusterPos[i, 0] != idx:
                clusterChange = True
            clusterPos[i,:] = idx, minD

        for i in range(k):
            nxtCLuster = dataset[np.nonzero(clusterPos[:, 0] == i)[0]]
            centers[i, :] = np.mean(nxtCLuster, axis=0)

        plt.title('this is iteration ' + str(iter), fontsize=10)
        iter += 1
        plt.scatter(dataset[:, 0], dataset[:, 1], c=clusterPos[:, 0], s = 3, marker = 'o')
        plt.scatter(centers[:, 0].A, centers[:, 1].A, c='red', s=100, marker='x')
        plt.scatter(centers_bak[:, 0].A, centers_bak[:, 1].A, c='blue', s=100, marker='x')
        plt.pause(1)
        if not clusterChange:
            plt.ioff()
            print(centers)
            print(centers_bak)
            plt.show()

    return centers, clusterPos



dataset, y = createData()
centers, clusterPos = kmeans(dataset, 3, 0)
