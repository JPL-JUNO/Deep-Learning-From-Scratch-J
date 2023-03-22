import sys
import os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
from common.optimizer import *

def f(x, y):
    return x ** 2 / 10.0 + y ** 2

def df(x, y):
    return x / 10.0, 2.0 * y

init_ops = (-7.0, 2.0)
params = {}
params['x'], params['y'] = init_ops[0], init_ops[1]

grads = {}
grads['x'], grads['y'] = 0, 0

optimizers = OrderedDict()
optimizers['SGD'] = SGD(lr=.95)
optimizers['Momentum'] = Momentum(lr=.1)
optimizers['AdaGrad'] = AdaGrad(lr=1.5)
optimizers['Adam'] = Adam(lr=.3)

idx = 0
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for key, ax in zip(optimizers.keys(), axes.ravel()):
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_ops[0], init_ops[1]
    
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        
        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)
    x = np.arange(-10, 10, .01)
    y = np.arange(-5, 5, .01)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    mask = Z > 7
    Z[mask] = 0
    ax.contour(X, Y, Z)
    ax.plot(x_history, y_history, 'ro-', label=key)
    ax.plot(0, 0, '+')
    ax.set_title(key)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)