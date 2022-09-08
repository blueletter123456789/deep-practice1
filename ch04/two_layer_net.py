import atexit
import sys, os

import numpy as np

sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, 
                weight_init_std=0.01):
        # initialize weight params.
        self.params = {}
        # weight of layer 1.
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # bias of layer 1.
        self.params['b1'] = np.zeros(hidden_size)
        # weight of layer 2.
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # bias of layer 2.
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """Execute predict

        Args:
            x (ndarray): Image data

        Returns:
            ndarray: output of neural network
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # layer 1.
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        # layer 2.
        a2 = np.dot(z1, W2) + b2
        y = soft_max(a2)

        return y
    
    def loss(self, x, t):
        """Calculate value of loss function

        Args:
            x (ndarray): image data
            t (ndarray): training data

        Returns:
            float: value of loss function
        """
        y = self.predict(x)

        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        """Calculate predict accuracy

        Args:
            x (ndarray): image data
            t (ndarray): training data

        Returns:
            float: value of accuracy
        """
        y = self.predict(x)
        # select maximum value of each array
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerial_gradient(self, x, t):
        """Calculate the gradient of weight and bias parameters

        Args:
            x (ndarray): image data
            t (ndarray): training data

        Returns:
            ndarray: gradient of params
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        # gradient of weight and bias each layer.
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = soft_max(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads
