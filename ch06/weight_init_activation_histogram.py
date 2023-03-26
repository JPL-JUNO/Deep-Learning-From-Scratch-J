import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from numpy import ndarray


def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))


def Relu(x: ndarray) -> ndarray:
    return np.maximum(x, 0)


def tanh(x: ndarray) -> ndarray:
    return np.tanh(x)


input_data = np.random.randn(1000, 100)  # 样本数、特征数
node_num = 100  # 每个隐藏层的节点数
hidden_layer_size = 5  # 隐藏层数
activations: Dict[int, ndarray] = {}  # 存储激活值


x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

    # w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * .01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
    a = np.dot(x, w)
    # z = sigmoid(a)
    z = Relu(a)
    # z = tanh(a)

    activations[i] = z

for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + '-layer')
    if i != 0:
        plt.yticks([], [])
    plt.hist(a.flatten(), bins=30, range=(0, 1))
plt.show()
