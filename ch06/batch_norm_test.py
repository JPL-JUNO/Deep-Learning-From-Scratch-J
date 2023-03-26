import sys
import os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 使用少量学习数据

x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = .01


def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(
        input_size=784, hidden_size_list=[100] * 5, output_size=10, weight_init_std=weight_init_std, use_batch_norm=True)
    network = MultiLayerNetExtend(
        input_size=784, hidden_size_list=[100] * 5, output_size=10, weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)
    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0
    for i in range(100_000_000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print('epoch: ' + str(epoch_cnt) + '|' +
                  str(train_acc) + '-' + str(bn_train_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break  # 直接跳出range循环
    return train_acc_list, bn_train_acc_list


weight_scale_list = np.around(np.logspace(0, -4, num=16), 4)
x = np.arange(max_epochs)

fig, axes = plt.subplots(2, 8, figsize=(32, 8))
for (i, w), ax in zip(enumerate(weight_scale_list), axes.ravel()):
    print('=========' + str(i + 1) + '/16' + '=========')
    train_acc_list, bn_train_acc_list = __train(w)
    ax.set_title('W=' + str(w))
    ax.plot([1, 2], [3, 4], label='a')
    ax.plot([1, 2], [5, 64], label='b')
    ax.plot(x, bn_train_acc_list, markevery=2)
    ax.plot(x, train_acc_list, '--', markevery=2)
    ax.set_ylim(0, 1)
    if i % 8:
        ax.set_yticks([])
    else:
        ax.set_ylabel('Accuracy')
    if i < 8:
        ax.set_xticks([])
    else:
        ax.set_xlabel('Epoch')


line, label = axes.ravel()[0].get_legend_handles_labels()

fig.legend(['Batch Normalization', 'Normal(without BatchNorm)'], ncol=2, bbox_to_anchor=(
    1, 1.05), bbox_transform=fig.transFigure, fontsize=15)
plt.tight_layout()
plt.show()
