from collections import OrderedDict
import sys, os

import numpy as np

sys.path.append(os.pardir)
from common.layers import *
from common.gradient import numerical_gradient

class MultiLayerNet:
    """Multilayer nueral network with total coupling
    """
    def __init__(self, input_size, hidden_size_list, output_size,
            activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        """Initializing multi layer network

        Args:
            input_size (int): Input size
            hidden_size_list (ndarray): List of the number of neurons in the hidden layer
            output_size (int): Output size
            activation (str, optional): Function name of the output layer. 
                Defaults to 'relu'.
            weight_init_std (str, optional): Specify standard deviation of weights (e.g. 0.01)
                If 'relu' or 'he' is specified, "Initial value of He" is set
                If 'sigmoid' or 'xavier' is specified, "Initial value of Xavier" is set. 
                Defaults to 'relu'.
            weight_decay_lambda (int, optional): Strength of Weight Decay (L2 norm). 
                Defaults to 0.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # initializing weight parameters
        self.__init_weight(weight_init_std)

        # create layers
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine'+str(idx)] = Affine(
                    self.params['W'+str(idx)], self.params['b'+str(idx)])
            self.layers['Activation_function'+str(idx)] = activation_layer[activation]()
        
        idx = self.hidden_layer_num + 1
        self.layers['Affine'+str(idx)] = Affine(
                self.params['W'+str(idx)], self.params['b'+str(idx)])
        
        self.last_layer = SoftmaxWithLoss()
    
    def __init_weight(self, weight_init_std):
        """Initializing weiht parameters

        Args:
            weight_init_std (str): Specify standard deviation of weights
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx-1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx-1])
            
            self.params['W'+str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b'+str(idx)] = np.zeros(all_size_list[idx])
    
    def predict(self, x):
        """Predicting in Affine and Activation layer

        Args:
            x (ndarray): args of affine or activation function

        Returns:
            ndarray: value of neural network
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

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            # To add L2 norm: 1/2*λ*W^2
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay
    
    def accuracy(self, x, t):
        """Calculate predict accuracy

        Args:
            x (ndarray): image data
            t (ndarray): training data

        Returns:
            float: value of accuracy
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """Calculate the gradient of weight and bias parameters

        Args:
            x (ndarray): image data
            t (ndarray): training data

        Returns:
            ndarray: gradient of params
        """
        loss_W = self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W'+str(idx)] = numerical_gradient(loss_W, self.params['W'+str(idx)])
            grads['b'*str(idx)] = numerical_gradient(loss_W, self.params['b'+str(idx)])
        
        return grads
    
    def gradient(self, x, t):
        """Calculate the gradient by backpropagation

        Args:
            x (ndarray): image data
            t (ndarray): training data

        Returns:
            ndarray: gradient of params
        """
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            # In back propagation λW is propagated.
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].dB

        return grads
