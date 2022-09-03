import sys
import os

import numpy as np

sys.path.append(os.pardir)
from dataset.mnist import load_mnist


def cross_entropy_error(y, t):
    """Cross entropy

    Args:
        y (List): output of neural network
        t (List): training data(numeric values)

    Returns:
        float: cross entropy loss
    """
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def cross_entropy_error_num(y, t):
    """Cross entropy

    When training data is given as labels (numerical values)

    Args:
        y (List): output of neural network
        t (List): training data(numeric values)

    Returns:
        float: cross entropy loss
    """
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True)
    # (x_train, t_train), (x_test, t_test) = load_mnist(
    #     normalize=True, one_hot_label=False)
    
    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # it is need processing by neural network...

    # Ex.
    t = np.array([0,0,1,0,0,0,0,0,0,0])
    y = np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])
    
    print(cross_entropy_error(y, t))

    y = np.array([0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0])
    t = np.array([2])
    print(cross_entropy_error_num(y, t))

