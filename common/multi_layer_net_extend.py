from collections import OrderedDict
import sys, os

import numpy as np

sys.path.append(os.pardir)
from common.layers import *
from common.gradient import numerical_gradient

class MultiLayerNetExtend:
    """Extended version of multilayer nueral network with total coupling
        - Added "Weight Decay", "Dropout" and "Batch Normalization" functions
    """
    def __init__(
            self, input_size, hidden_size_list, output_size,
            activation='relu', weight_init_std='relu', weight_decay_lambda=0,
            use_dropout=False, dropout_ration=0.5, use_batchnorm=False):
        """Initializing each parameter

        Args:
            input_size (int): input size
            hidden_size_list (List): layer size of hidden layer
            output_size (int): output size
            activation (str, optional): Function name of the output layer. 
                Defaults to 'relu'.
            weight_init_std (str, optional): Specify standard deviation of weights (e.g. 0.01)
                If 'relu' or 'he' is specified, "Initial value of He" is set
                If 'sigmoid' or 'xavier' is specified, "Initial value of Xavier" is set. 
                Defaults to 'relu'.
            weight_decay_lambda (int, optional): Strength of Weight Decay (L2 norm). 
                Defaults to 0.
            use_dropout (bool, optional): using Dropout. Defaults to False.
            dropout_reation (float, optional): rate value of Dropout. Defaults to 0.5.
            use_batchnorm (bool, optional): using Batchnormalization. Defaults to False.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        self.__init_weight(weight_init_std)

        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine'+str(idx)] = Affine(
                    self.params['W'+str(idx)], self.params['b'+str(idx)])
            if self.use_batchnorm:
                # yi ← γxi + β
                self.params['gamma'+str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta'+str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm'+str(idx)] = BatchNormalization(
                        self.params['gamma'+str(idx)], self.params['beta'+str(idx)])
            
            self.layers['Activation_function'+str(idx)] = activation_layer[activation]()

            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)
        
        idx = self.hidden_layer_num + 1
        self.layers['Affine'+str(idx)] = Affine(
                self.params['W'+str(idx)], self.params['b'+str(idx)])
        
        self.last_layer = SoftmaxWithLoss()
    
    def __init_weight(self, weight_init_std):
        """Initializing weight value

        Args:
            weight_init_std (str): Standard deviation of weights
                'relu' or 'he': (initialize weight value) * (1 / √n)
                'sigmoid' or 'xavier': (initialize weight value) * (2 / √n)
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx-1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx-1])
            self.params['W'+str(idx)] = scale * np.random.randn(
                        all_size_list[idx-1],all_size_list[idx])
            self.params['b'+str(idx)] = np.zeros(all_size_list[idx])
    
    def predict(self, x, train_flg=False):
        """Predicting in Affine and Activation layer

        Args:
            x (ndarray): args of affine or activation function

        Returns:
            ndarray: value of neural network
        """
        for key, layer in self.layers.items():
            if 'Dropout' in key or 'BatchNorm' in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        
        return x
    
    def loss(self, x, t, train_flg=False):
        """Calculate value of loss function

        Args:
            x (ndarray): image data
            t (ndarray): training data

        Returns:
            float: value of loss function
        """
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay
    
    def accuracy(self, x, t):
        """Calculate predict accuracy

        Args:
            x (ndarray): image data
            t (ndarray): training data

        Returns:
            float: value of accuracy
        """
        y = self.predict(x, train_flg=False)
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
        loss_W = lambda W: self.loss(x, t, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W'+str(idx)] = numerical_gradient(loss_W, self.params['W'+str(idx)])
            grads['b'+str(idx)] = numerical_gradient(loss_W, self.params['b'+str(idx)])

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma'+str(idx)] = numerical_gradient(loss_W, self.params['gamma'+str(idx)])
                grads['beta'+str(idx)] = numerical_gradient(loss_W, self.params['beta'+str(idx)])
        
        return grads
    
    def gradient(self, x, t):
        """Calculate the gradient by backpropagation

        Args:
            x (ndarray): image data
            t (ndarray): training data

        Returns:
            ndarray: gradient of params
        """
        self.loss(x, t, train_flg=True)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].dB

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads
