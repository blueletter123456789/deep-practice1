import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# 'x' has 1000x100 data
x = np.random.randn(1000, 100)
# hidden node size
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    
    # value become 0, Vanishing gradient problem is occurred
    w = np.random.randn(node_num, node_num) * 1

    # value become 0.5. Neurons become incapable of diverse expressions.
    # w = np.random.randn(node_num, node_num) * 0.01

    # Xavier's initial value
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)

    # He's initial value
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    z = np.dot(x, w)
    a = sigmoid(z)
    # a = tanh(z)
    # a = Relu(z)

    activations[i] = a

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title('{}-layer'.format(str(i+1)))
    plt.hist(a.flatten(), 30, range=(0, 1))

    # If using tahn(), take a range from -1 to +1.
    # plt.hist(a.flatten(), 30, range=(-1, 1))

plt.show()
