import numpy as np


def softmax(x):
    # 感觉没有转置的必要
    # 完全可以用axis=1替代
    if x.ndim == 2:
        x = x.T
        x = x - x.np.max(x, axis=0)
        y = np.np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identify_function(x):
    return x


def relu(x):
    return np.maximum(0, x)
