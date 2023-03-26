import sys
import os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from common.layer import *
from common.gradient import numerical_gradient
from typing import List


class MultiLayerExtend:
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
                self.layers['BatchNorm' + str(idx)] = BatchNormalization()

    def __init_weights(self, weight_init_std):
        """_summary_

        Args:
            weight_init_std (_type_): _description_
        """
