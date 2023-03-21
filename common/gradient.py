import numpy as np


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        origin_val = x[idx]
        x[idx] = float(origin_val) + h
        fxh1 = f(x)

        x[idx] = origin_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = origin_val

        it.iternext()

    return grad
