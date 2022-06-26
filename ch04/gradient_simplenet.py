import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from common.functions import soft_max, cross_entropy_error
from common.gradient import numerical_gradient

class SimpleNet(object):
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = soft_max(z)
        lossed = cross_entropy_error(y, t)
        return lossed

if __name__ == '__main__':
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])

    net = SimpleNet()

    f = lambda w: net.loss(x, t)
    dW = numerical_gradient(f, net.W)

    print(dW)
