import numpy as np
import matplotlib.pyplot as plt

from simple_convnet import SimpleConvNet


def filter_show(filters, nx=10, margin=3, scale=10):
    """_summary_

    Args:
        filters (_type_): _description_
        nx (int, optional): _description_. Defaults to 8.
        margin (int, optional): _description_. Defaults to 3.
        scale (int, optional): _description_. Defaults to 10.
    """

    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0.5,
                        top=1, hspace=0, wspace=.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i + 1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


network = SimpleConvNet()
filter_show(network.params['W1'])

network.load_params('params.pkl')
filter_show(network.params['W1'])
