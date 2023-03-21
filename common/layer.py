import numpy as np
from numpy import ndarray
from common.function import softmax
from common.function import cross_entropy_error


class Relu:
    def __init__(self) -> None:
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self) -> None:
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        ones = np.ones_like(self.out)
        dx = dout * (ones - self.out) * self.out
        return dx


class Affine:
    '''
    实现一次仿射变换
    '''

    def __init__(self, W, b) -> None:
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x: ndarray) -> ndarray:
        self.x = x
        out = np.dot(x, self.W)
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据的真实标签，one-hot vector

    def forward(self, x, t) -> float:
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout) -> ndarray:
        batch_size = self.t.shape[0]

        # 因为在计算交叉熵时，自带了一个系数，因此反向传播\frac{\partial L}{\partial y}=\frac{1}{batch_size}一开始就不是1，而是1/batch_size
        dx = (self.y - self.t) / batch_size
        return dx
