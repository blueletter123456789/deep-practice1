import numpy as np

def cross_entropy_error(y, t):
    """Cross Entropy Error

    Args:
        y (List): output of neural network
        t (List): true label

    Returns:
        float: Cross-Entropy Error
    """
    # if np.log(0), then it is calculated as -inf
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

if __name__ == '__main__':
    t = np.array([0,0,1,0,0,0,0,0,0,0])
    y = np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])
    
    print(cross_entropy_error(y, t))

    y = np.array([0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0])

    print(cross_entropy_error(y, t))

