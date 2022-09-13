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

class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        self.running_mean = running_mean
        self.running_var = running_var  
        
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
    
    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
    
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out
    
    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx
