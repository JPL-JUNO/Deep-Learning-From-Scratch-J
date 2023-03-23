import sys
import os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from common.optimizer import *
from common.multi_layer_net import MultiLayerNet
from common.util import smooth_curve

(x_train, t_train), (y_test, t_test) = load_mnist(normalize=True)


train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2_000

optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10
    )

    train_loss[key] = []

for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print('=======' + 'iteration: ' + str(i) + '=======')
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ':' + str(loss))
