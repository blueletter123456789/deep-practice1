import numpy as np


def sum_squared_error(y, t):
    """the Sum of Squared Error(SSE)

    The loss function is an indicator of poor performance.

    Args:
        y (List): output of neural network
        t (List): training data

    Returns:
        float: SSE
    """
    return 0.5 * np.sum((y - t)**2)

if __name__ == '__main__':
    t = np.array([0,0,1,0,0,0,0,0,0,0])
    y = np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])
    
    print(sum_squared_error(y, t))

    y = np.array([0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0])

    print(sum_squared_error(y, t))
