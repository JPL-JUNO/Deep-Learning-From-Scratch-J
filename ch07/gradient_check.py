import numpy as np
from simple_convnet import SimpleConvNet

network = SimpleConvNet(input_dim=(1, 10, 10),
                        conv_params={'filter_num': 10,
                                     'filter_size': 3, 'pad': 0, 'stride': 1},
                        hidden_size=10, output_size=10, weight_init_std=.01)
X = np.random.rand(100).reshape(1, 1, 10, 10)
T = np.array([1]).reshape(1, 1)

grad_numerical = network.numerical_gradient(X, T)
grad_backprop = network.gradient(X, T)

for key, val in grad_numerical.items():
    print(key, np.abs(grad_numerical[key] - grad_backprop[key]).mean())
