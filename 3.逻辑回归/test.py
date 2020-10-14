import numpy as np

import matplotlib.pyplot as plt

def load_data():
    X1 = np.random.rand(50, 1)/2 + 0.5
    X2 = np.random.rand(50, 1)/2 + 0.5
    y = X1 + X2
    Y = y > 1.4
    Y = Y + 0
    x = np.c_[np.ones(len(Y)), X1, X2]
    return x, Y


def sigmod(x):
    return 1.0 / (1 + np.exp(-x))

#全量梯度下降
def gradAscent(data, classLabels):
    X = np.mat(data)
    Y = np.mat(classLabels)
    m, n = np.shape(data) 
    alpha = 0.01
    iterations = 10000
    wei = np.ones((n, 1))
    for i in range(iterations):
        y_hat = sigmod(X * wei)
        error = y_hat - Y
        wei = wei - alpha * np.dot(X.T, error)
    return wei

#随机梯度下降
def stocGradAscent(data, classLabels, iter_cnt):
    X = np.mat(data)
    Y = np.mat(classLabels)
    m, n = np.shape(data) 
    wei = np.ones((n, 1))
    for i in range(iter_cnt):
        for j in range(m):
            alpha = 4 / (1.0 + i + j) + 0.01
            randIdx = int(np.random.uniform(0, m))
            y_hat = sigmod(np.sum(X[randIdx] * wei))
            error = y_hat - Y[randIdx]
            wei = wei - alpha * np.dot(X[randIdx].T, error)
    return wei


def plotPic():
    dataMat, labelMat = load_data()
    #getA()矩阵转换为数组
    weis = gradAscent(dataMat, labelMat).getA()
    weis_stoc = stocGradAscent(dataMat, labelMat, 2000).getA()
    m = np.shape(dataMat)[0]
    xc1 = []
    yc1 = []
    xc2 = []
    yc2 = []
    for i in range(m):
        if int(labelMat[i]) == 1:
            xc1.append(dataMat[i][1])
            yc1.append(dataMat[i][2])
        else:
            xc2.append(dataMat[i][1])
            yc2.append(dataMat[i][2])

    plt.scatter(xc1, yc1, s=30, c='red', marker='s')
    plt.scatter(xc2, yc2, s=30, c='green')

    x = np.arange(0.5, 1, 0.1)
    y = (-weis[0] - weis[1] * x) / weis[2]
    y_stoc = (-weis_stoc[0] - weis_stoc[1] * x) / weis_stoc[2]
    print(weis)
    print(weis_stoc)
    plt.plot(x, y, 'b-')
    plt.plot(x, y_stoc, 'r-')
    plt.show() 



plotPic()