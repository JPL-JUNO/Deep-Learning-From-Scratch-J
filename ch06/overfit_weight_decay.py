import os
import sys

sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 减少训练数据，使得过拟合
x_train = x_train[:300]
t_train = t_train[:300]

# weight_decay_lambda = 0
weight_decay_lambda = 0.1


network = MultiLayerNet(input_size=784, hidden_size_list=[100] * 6,
                        output_size=10,
                        weight_decay_lambda=weight_decay_lambda)

optimizer = SGD(lr=.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(100_000_000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print('epoch: ' + str(epoch_cnt) + ', train accuracy: ' +
              str(train_acc) + ', test accuracy: ' + str(test_acc))
        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xlim(0, 200)
plt.ylim(0, 1)
plt.legend()
plt.show()
