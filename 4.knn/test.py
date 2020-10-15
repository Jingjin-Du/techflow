import random 
import numpy as np
from collections import Counter

def classify(x, dataset, labels, K):
    x, dataset = np.array(x), np.array(dataset)
    dis = np.sum((x - dataset) ** 2, axis = 1)
    topKIdices = np.argsort(dis)[:K]
    labels = np.array(labels)
    counter = Counter(labels[topKIdices])
    return counter.most_common(1)[0][0]


def create_data_set():
    dataset = np.array([[0.5, 0], [0, 0.5], [1.5, 1], [1, 1.5]])
    labels = ['A', 'A', 'B', 'B']
    return dataset, labels


dataset, labels = create_data_set()
predict = classify([0, 0], dataset, labels, 3)
print(predict)