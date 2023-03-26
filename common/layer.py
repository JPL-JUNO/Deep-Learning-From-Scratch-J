import numpy as np
from numpy import ndarray
from common.function import softmax
from common.function import sigmoid
from common.function import cross_entropy_error


class Relu:
    def __init__(self) -> None:
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
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
        out = sigmoid(x)
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
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x: ndarray) -> ndarray:
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
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
        # # 因为在计算交叉熵时，自带了一个系数，因此反向传播\frac{\partial L}{\partial y}=\frac{1}{batch_size}一开始就不是1，而是1/batch_size
        # dx = (self.y - self.t) / batch_size

        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma, beta, momentum=.9, running_mean=None, running_var=None) -> None:
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv 层的情况下为四维，全连接层的情况下维2维

        # 测试时使用的平均值方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        """_summary_

        Args:
            x (_type_): _description_
            train_flg (bool, optional): _description_. Defaults to True.
        """
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
        out = self.__forward(x, train_flg)
        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        """_summary_

        Args:
            x (_type_): _description_
            train_flg (_type_): _description_
        """
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
        if train_flg:  # 进行标准化
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * \
                self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * \
                self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 1e-7)))
        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        """_summary_

        Args:
            dout (_type_): _description_
        """
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)
        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std

        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = .5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """

    def __init__(self, dropout_ratio: float = .5) -> None:
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg: bool = True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
