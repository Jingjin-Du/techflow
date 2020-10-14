'''
公式直接求解
theta = (X.T * X)^-1 * X.T * y
梯度下降求解
'''
import numpy as np
import matplotlib.pyplot as plt


def get_theta(x, y):
    m = len(y)
    x = np.c_[np.ones(m).T, x]
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
    print("theta by equation:")
    print(theta)
    return theta


def get_theta_grad(x, y):
    eta = 0.1
    n_iterations = 1000
    m = len(y)
    theta = np.random.randn(2, 1)
    x = np.c_[np.ones(100).T, x]
    for i in range(n_iterations):
        gradients = 1/m * x.T.dot(np.dot(x, theta) - y)
        theta = theta - eta * gradients
    print("theta by gradients")
    print(theta)
    return theta


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
plt.scatter(X, y)

theta = get_theta(X, y)
theta_grad = get_theta_grad(X, y)

X_new = np.array([[0], [1]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta)
plt.plot(X_new, y_predict, 'r-')

X_new = np.array([[1], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict_grad = X_new_b.dot(theta_grad)
plt.plot(X_new, y_predict_grad, 'b-')


plt.show()