import numpy as np

# Sample code
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identify_function(X):
    return X

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = identify_function(a3)

    return y

if __name__ == '__main__':
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def identify_function(X):
#     return X

# def neural(X, W, B, identify=False):
#     A = np.dot(X, W) + B
#     # print(A)
#     if identify:
#         return identify_function(A)
#     return sigmoid(A)

# if __name__ == '__main__':
#     # 入力層から第１層
#     X1 = np.array([1, 0.5])
#     W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
#     B1 = np.array([0.1, 0.2, 0.3])
#     Z1 = neural(X1, W1, B1)
#     print(Z1)

#     # 第１層から第２層
#     W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
#     B2 = np.array([0.1, 0.2])
#     Z2 = neural(Z1, W2, B2)
#     print(Z2)

#     # 第２層から出力層
#     """
#     出力層の活性化関数は解く問題により異なる
#         回帰問題：恒等関数
#         ２クラス分類問題：シグモイド関数
#         多クラス分類：ソフトマックス関数
#     """
#     W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
#     B3 = np.array([0.1, 0.2])
#     Y = neural(Z2, W3, B3, True)
#     print(Y)


