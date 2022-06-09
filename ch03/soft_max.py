import numpy as np

# def soft_max(a):
#     """
#     e^xのためxの値が大きい場合にオーバーフローを引き起こす
#     """
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a

#     return y

def soft_max(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

if __name__ == '__main__':
    a = np.array([0.3, 2.9, 4.0])
    print(soft_max(a))
    
    a2 = np.array([1010, 1000, 990])
    print(soft_max(a2))
