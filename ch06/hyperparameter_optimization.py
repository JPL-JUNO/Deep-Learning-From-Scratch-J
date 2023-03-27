import sys
import os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer
from typing import List, Tuple


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 减少数据量，加快学习
x_train = x_train[:500]
t_train = t_train[:500]

validation_rate = .2
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)

# 前面的一部分（20%）留出来进行验证调参
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(lr, weight_decay, epochs=50):
    network = MultiLayerNet(input_size=784, hidden_size_list=[100] * 6, output_size=10,
                            weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epochs, mini_batch_size=100,
                      optimizer='SGD', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()
    return trainer.test_acc_list, trainer.train_acc_list


optimization_trail = 100
results_val = {}
results_train = {}

for _ in range(optimization_trail):
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)
    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print('Validation Accuracy: {:.3f} | lr:{}, weight_decay: {}'.format(
        val_acc_list[-1], lr, weight_decay))
    key = 'lr: {}, weight_decay: {}'.format(lr, weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

sorted_results_val: List[Tuple[str, list]] = sorted(
    results_val.items(), key=lambda x: x[1][-1], reverse=True)
fig, axes = plt.subplots(2, 8, figsize=(32, 8))
for idx, ((key, val_acc_list), ax) in enumerate(zip(sorted_results_val, axes.ravel())):
    if idx % 8:
        ax.set_yticks([])
    ax.set_title('Best-' + str(idx + 1))
    ax.plot(np.arange(len(val_acc_list)), val_acc_list)
    ax.plot(np.arange(len(val_acc_list)), results_train[key], '--')

plt.show()
