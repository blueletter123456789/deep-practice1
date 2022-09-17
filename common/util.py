import numpy as np


def smooth_curve(x):
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """Shuffle the data set

    Args:
        x (ndarray): training data
        t (ndarray): training data(teacher data)

    Returns:
        ndarray: shuffled data
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """Expand input data in two dimensions

    Args:
        input_data (ndarray): input data(number of data, channel, height, width)
        filter_h (int): height of filter
        filter_w (int): width of filter
        stride (int, optional): stride size. Defaults to 1.
        pad (int, optional): padding size. Defaults to 0.

    Returns:
        ndarray: two dimensional array
    """
    N, C, H, W = input_data.shape
    # Calculate output data size
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # Padding for each image data. "constant" means zero padding
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    # Stores the filter's area
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            # Store each filter by row
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    # Shaped to be (NOhOw, CFhFw)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """Formatting from a 2-dimensional array to the dimensions of the input data

    Args:
        col (ndarray): two dimensional array
        input_shape (tuple): shape of input data
        filter_h (int): height of filter
        filter_w (int): width of filter
        stride (int, optional): Number of filters stride. Defaults to 1.
        pad (int, optional): Number of input data padding. Defaults to 0.

    Returns:
        ndarray: input data(number of array, channel, height, width)
    """
    N, C, H, W = input_shape
    out_h = int((H + 2*pad - filter_h) / stride) + 1
    out_w = int((W + 2*pad - filter_w) / stride) + 1
    # (NOhOw, CFhFw) formatted as (N, C, Fh, Fw, Oh, Ow)
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
    # return as (N, C, H, W)
    return img[:, :, pad:H+pad, pad:W+pad]
