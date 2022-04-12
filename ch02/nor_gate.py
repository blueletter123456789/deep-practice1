import numpy as np

def NOR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.3
    tmp = np.sum(x*w)+b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    x = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for x1, x2 in x:
        print(NOR(x1, x2))
