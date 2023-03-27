import os
import sys
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

x_train = x_train[:300]
t_train = t_train[:300]


use_dropout = True
dropout_ratio = 0

network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100] * 6,
                              output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=301, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr': .01}, verbose=False)
trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

x = np.arange(len(train_acc_list))

plt.plot(x, train_acc_list, marker='o', label='Train', markevery=10)
plt.plot(x, test_acc_list, marker='o', label='Train', markevery=10)
plt.title('Dropout=' + str(dropout_ratio))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)
plt.legend()
plt.show()
