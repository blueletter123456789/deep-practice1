import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from common.functions import soft_max, cross_entropy_error
from common.gradient import numerical_gradient

class SimpleNet(object):
    def __init__(self):
        # Initialize with normal distribution
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        
        # output layer
        y = soft_max(z)
        
        # Calculate value of loss function using with Cross-Entropy
        lossed = cross_entropy_error(y, t)

        return lossed

if __name__ == '__main__':
    # input data
    x = np.array([0.6, 0.9])
    # training data
    t = np.array([0, 0, 1])

    net = SimpleNet()

    # "w" is just a dummy for using numerical_gradient()
    f = lambda w: net.loss(x, t)

    # Calculate the gradient of the loss function.
    dW = numerical_gradient(f, net.W)

    print(dW)
