import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        '''
        进行初始化时的参数 lr 表示learning rate（学习率）
        '''
        self.lr = lr

    def update(self, params, grads):
        '''
        参数 params和 grads 是字典型变量，按 params['W1']、grads['W1']的形式，分别保
        存了权重参数和它们的梯度。
        '''
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    '''
    '''

    def __init__(self, lr=.01, momentum: float = .9) -> None:
        '''
        实例变量 v会保存物体的速度。
        '''
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        '''
        初始化时，v中什么都不保存，但当第
        一次调用 update()时，v会以字典型变量的形式保存与参数结构相同的数据。
        '''
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    '''
    '''

    def __init__(self, lr=.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        '''
        '''
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    '''
    http://arxiv.org/abs/1412.6980v8
    '''

    def __init__(self, lr=.001, beta1=.9, beta2=.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter))
        for key in params.keys():
            self.m[key] += (1-self.beta1)*(grads[key]-self.m[key])
