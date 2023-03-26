import sys
import os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from common.layer import *
from common.gradient import numerical_gradient
from typing import List


class MultiLayerNetExtend:
    """
    拓展版的全连接多层神经网络
    具有权重衰减（weight decay），dropout, batch norm
    """

    def __init__(self, input_size: int, hidden_size_list: List[int], output_size: int,
                 activation: str = 'relu', weight_init_std: str = 'relu', weight_decay_lambda: float = 0,
                 use_dropout: bool = False, dropout_ration: float = .5, use_batch_norm: bool = False) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batch_norm = use_batch_norm
        self.params = {}

        self.__init_weights(weight_init_std)
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu, }
        self.layers = OrderedDict()

        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])

            if self.use_batch_norm:
                # 参数gamma和beta的长度应该等于当层神经元的节点数
                self.params['gamma' +
                            str(idx)] = np.ones(hidden_size_list[idx - 1])
                self.params['beta' +
                            str(idx)] = np.zeros(hidden_size_list[idx - 1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(
                    self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
            self.layers['Activation_function' +
                        str(idx)] = activation_layer[activation]()
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])
        self.last_layer = SoftmaxWithLoss()

    def __init_weights(self, weight_init_std):
        """设定权重的初始值

        Args:
            weight_init_std (_type_): _description_
        """

        all_size_list = [self.input_size] + \
            self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('xavier', 'sigmoid'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            self.params['W' + str(idx)] = scale * \
                np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = scale * \
                np.random.randn(all_size_list[idx])

    def predict(self, x, train_flg=False):
        """_summary_

        Args:
            x (_type_): _description_
            train_flg (bool, optional): _description_. Defaults to False.
        """
        for key, layer in self.layers.items():
            # 只对dropout和batch norm层做，这两个类的前向传播方法确实有两个参数
            if 'Dropout' in key or 'BatchNorm' in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        """_summary_

        Args:
            x (_type_): _description_
            t (_type_): _description_
            train_flg (bool, optional): _description_. Defaults to False.
        """
        y = self.predict(x, train_flg)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += .5 * self.weight_decay_lambda * np.sum(W ** 2)
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, X, T):
        """_summary_

        Args:
            x (_type_): _description_
            T (_type_): _description_
        """
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1:
            T.np.argmax(T, axis=1)
        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def numerical_gradient(self, X, T):
        """_summary_

        Args:
            X (_type_): _description_
            T (_type_): _description_
        """
        def loss_W(W): return self.loss(X, T, train_flg=False)
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(loss_W,
                                                       self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W,
                                                       self.params['b' + str(idx)])
            if self.use_batch_norm and idx != (self.hidden_layer_num + 1):
                grads['gamma' + str(idx)] = numerical_gradient(loss_W,
                                                               self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W,
                                                              self.params['beta' + str(idx)])
        return grads

    def gradient(self, x, t):
        """_summary_

        Args:
            x (_type_): _description_
            t (_type_): _description_
        """
        self.loss(x, t, train_flg=True)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + \
                self.weight_decay_lambda * self.params['W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db
            if self.use_batch_norm and idx != self.hidden_layer_num + 1:
                grads['gamma' +
                      str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' +
                      str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta
        return grads
