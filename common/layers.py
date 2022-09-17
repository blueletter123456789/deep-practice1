from turtle import forward
import numpy as np

from common.functions import *
from common.util import im2col, col2im

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
            # Create an array of the same shape as x. (True of False)
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        # Only pass through what flows in forward propagation.
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

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        """Initialize convolutional layer

        Args:
            W (ndarray): array of filter
            b (ndarray): array of bias
            stride (int, optional): Number of filters stride. Defaults to 1.
            pad (int, optional): Number of input data padding. Defaults to 0.
        """
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # intermediate data
        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.dB = None
    
    def forward(self, x):
        """Forwarding in convolutional layer

        Args:
            x (ndarray): image data

        Returns:
            ndarray: output data
        """
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        out_h = int((H + 2*self.pad - FH)/self.stride) + 1
        out_w = int((W + 2*self.pad - FW)/self.stride) + 1

        col = im2col(x, FH, FW, self.stride, self.pad)
        # Arrange filters by columns.
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        # (N, H, W, C) → (N, C, H, W)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out
    
    def backward(self, dout):
        """Backward in convolution layer

        Args:
            dout (ndarray): Differentiation of y

        Returns:
            ndarray: Differentiation of x
        """
        FN, C, FH, FW = self.W.shape
        # (N, FN, Oh, Ow) → (NOhOw, FN)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # Number of each element of the gradient added (1, FN)
        self.dB = np.sum(dout, axis=0)

        # Differentiation of filter
        # (CFhFw, NOhOw) * (NOhOw, FN) = (CFhFw, Fn)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # Differentiation of input data
        # (NOhOw, FN) * (FN, CFhFw) = (NOhOw, CFhFw)
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None
    
    def forward(self, x):
        N, C, H, W = x.shape

        out_h = int((H - self.pool_h) / self.stride) + 1
        out_w = int((W - self.pool_w) / self.stride) + 1

        # (N C, H, W) → (NOhOw, CPhPw)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # (NOhOw, CPhPw) → (NOhOwC, PhPw)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # Get the index of the maximum value of each row. (NOhOw, )
        arg_max = np.argmax(col, axis=1)
        # Get the maximum value of each row. (NOhOw, )
        out = np.max(col, axis=1)

        # (NOhOwC, ) → (N, C, Oh, Ow)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out
    
    def backward(self, dout):
        # input shape is (N, C, Oh, Ow)
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w

        # crate colsize (NCOhOw, PhPw)
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        # (N, Oh, Ow, C, PhPw) → (NOhOw, CPhPw)
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
    