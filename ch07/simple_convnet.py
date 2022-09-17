from collections import OrderedDict
import sys, os
import pickle

import numpy as np

sys.path.append(os.pardir)
from common.layers import *
from common.gradient import numerical_gradient


class SimpleConvNet:
    """Simple convolution network
        conv - relu - pool - affine - relu - affine - softmax
    """
    def __init__(self, 
        input_dim=(1, 28, 28), 
        conv_param={
            'filter_num': 30, 
            'filter_size': 5,
            'pad': 0,
            'stride': 1},
        hidden_size=100,
        output_size=10,
        weight_std_init=0.01
    ):
        """Initialize Simple convolution network

        Args:
            input_dim (tuple): Dimensions of input data. (channel, height, width)
            conv_param (dict): Hyper parameters of convolution layer
                filter_num: number of filter
                filter_size: filter size
                pad: padding size
                stride: stride size
            hidden_size (int): Number of neurons in hidden layer
            output_size (int): Number of neurons in output layer
            weight_std_init (float): Standard deviation of weights during initialization
        """
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size + 2*filter_pad - filter_size) // filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.params = {}
        self.params['W1'] = weight_std_init * np.random.randn(
                filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_std_init * np.random.randn(
            pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_std_init * np.random.randn(
            hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()
    
    def predict(self, x):
        """Execute predict

        Args:
            x (ndarray): Image data

        Returns:
            ndarray: output of neural network
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    def loss(self, x, t):
        """Calculate value of loss function

        Args:
            x (ndarray): image data
            t (ndarray): training data

        Returns:
            float: value of loss function
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t, batch_size=100):
        """Calculate predict accuracy

        Args:
            x (ndarray): image data
            t (ndarray): training data
            batch_size (int): batch sizes

        Returns:
            float: value of accuracy
        """
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        acc = 0.0

        for i in range(x.shape[0]//batch_size):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]

            y = self.predict(tx)
            y = np.argmax(y, axis=1)

            acc += np.sum(y == tt)
        
        return acc / x.shape[0]
    
    def numerical_gradient(self, x, t):
        """Calculate the gradient of weight and bias parameters

        Args:
            x (ndarray): image data
            t (ndarray): training data

        Returns:
            ndarray: gradient of params
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W'+str(idx)] = numerical_gradient(loss_W, self.params['W'+str(idx)])
            grads['b'+str(idx)] = numerical_gradient(loss_W, self.params['b'+str(idx)])
        
        return grads
    
    def gradient(self, x, t):
        """Calculating the gradient using back propagation.

        Args:
            x (ndarray): input data
            t (ndarray): training data

        Returns:
            ndarray: gradient
        """
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].dB
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].dB
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].dB

        return grads
    
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
