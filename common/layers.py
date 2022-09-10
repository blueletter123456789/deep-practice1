import numpy as np

from common.functions import *

class Relu:
    def __init__(self):
        """initialize RELU class

        mask (ndarray[bool])
        """
        self.mask = None
    
    def forward(self, x):
        """Forward in a RELU layer

        Args:
            x (ndarray): Value of x

        Returns:
            ndarray: Result of RELU
        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out
    
    def backward(self, dout):
        """Backward in a RELU layer

        Args:
            dout (ndarray): Differentiation of y

        Returns:
            ndarray: Differentiation of x
        """
        dout[self.mask] = 0

        dx = dout

        return dx

class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        """Forward in a Sigmoid layer

        Args:
            x (ndarray): Value of x

        Returns:
            ndarray: Result of sigmoid
        """
        # out = 1 / (1 + np.exp(-x))
        out = sigmoid(x)
        self.out = out

        return out
    
    def backward(self, dout):
        """Backward in a Sigmoid layer

        Args:
            dout (ndarray): Differentiation of y

        Returns:
            ndarray: Differentiation of x
        """
        dx = dout * (1 - self.out) * self.out

        return dx

class Affine:
    def __init__(self, W, B):
        """Initialize affine layer

        Args:
            W (ndarray): weight parameter
            B (ndarray): bias parameter
        """
        self.W = W
        self.B = B
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.dB = None
    
    def forward(self, x):
        """Forward in affine layer

        Args:
            x (ndarray): Value of x

        Returns:
            ndarray: Add the bias to the product of the matrices
        """
        # supporting tensor
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(x, self.W) + self.B

        return out
    
    def backward(self, dout):
        """Backward in affine layer

        Args:
            dout (ndarray): Differentiation of y

        Returns:
            ndarray: Differentiation of x
        """
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.dB = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        # value of loss function
        self.loss = None
        # output of softmax
        self.y = None
        # training data
        self.t = None
    
    def forward(self, x, t):
        """Execute output layer and loss function

        Args:
            x (ndarray): output of neural network
            t (ndarray): training data

        Returns:
            float: Value of loss function
        """
        self.t = t

        # Normalization of output
        self.y = soft_max(x)

        # loss function
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss
    
    def backward(self, dout=1):
        """

        Args:
            dout (int, optional): Defaults to 1.

        Returns:
            ndarray: Error of each value
        """
        batch_size = self.t.shape[0]

        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx
