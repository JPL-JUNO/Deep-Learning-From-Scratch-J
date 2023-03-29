import numpy as np
from numpy import ndarray


def smooth_curve(x):
    '''

    '''

    window_len = 11
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 1)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[5:len(y) - 5]


def shuffle_dataset(x, t):
    """打乱数据集

    Args:
        x (_type_): 训练数据
        t (_type_): 标签数据
    """

    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]

    return x, t


def im2col(input_data: ndarray, filter_h: int, filter_w: int, stride: int = 1, pad: int = 0) -> ndarray:
    """将图像数据转为列数据（四维转二维）

    Args:
        input_data (ndarray): 由(num, channel, height, width)的四维数组构成的输入数据
        filter_h (int): 过滤器高
        filter_w (int): 过滤器长
        stride (int, optional): 步幅. Defaults to 1.
        pad (int, optional): 填充. Defaults to 0.
    """
    N, C, H, W = input_data.shape

    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    # 对原始的数据进行填充，在每个数据的每个通道上，高和宽填充 pad 个0
    img = np.pad(input_data, [(0, 0), (0, 0),
                 (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride: int = 1, pad: int = 0):
    """_summary_

    Args:
        col (_type_): _description_
        input_shape (_type_): _description_
        filter_h (_type_): _description_
        filter_w (_type_): _description_
        stride (int, optional): _description_. Defaults to 1.
        pad (int, optional): _description_. Defaults to 0.
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h,
                      filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    return img[:, :, pad:H + pad, pad:W + pad]
