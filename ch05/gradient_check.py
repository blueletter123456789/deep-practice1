import sys, os

import numpy as np

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

x_batch = x_train[:3]
t_batch = t_train[:3]

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_numerical[key] - grad_backprop[key]))
    print('{}: {}'.format(key, diff))
