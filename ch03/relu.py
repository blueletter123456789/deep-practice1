import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

if __name__ == '__main__':
    x = np.arange(-4.0, 4.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 4.1)
    plt.show()
