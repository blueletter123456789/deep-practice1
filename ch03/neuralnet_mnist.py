import os
import pickle
import sys

import numpy as np

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def soft_max(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = soft_max(a3)

    return y

if __name__ == '__main__':
    x, t = get_data()
    network = init_network()

    accurancy_cnt = 0
    accurancy_lst = [[0]*2 for _ in range(10)]
    for i, tgt in enumerate(x):
        y = predict(network, tgt)
        p = np.argmax(y)
        accurancy_lst[t[i]][1] += 1
        if p == t[i]:
            accurancy_cnt += 1
            accurancy_lst[t[i]][0] += 1
    
    print('Accurancy: ', accurancy_cnt / len(x))
    for i, rst in enumerate(accurancy_lst):
        print('{}: {}'.format(i, rst[0]/rst[1]))

