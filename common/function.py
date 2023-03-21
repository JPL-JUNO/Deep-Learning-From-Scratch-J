import numpy as np
from numpy import ndarray


def softmax(x):
    # 感觉没有转置的必要
    # 完全可以用axis=1替代
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identify_function(x):
    return x


def relu(x):
    return np.maximum(0, x)


def cross_entropy_error(y: ndarray, t: ndarray):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
