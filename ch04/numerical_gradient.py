import numpy as np

def function_2(x):
    # smaple function
    # f(x0, x1) = x0^2 + x1^2
    return np.sum(x**2)

def numeric_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        # calculate forward difference
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # calculate backward dirrerence
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


if __name__ == '__main__':
    x1 = np.array([3.0, 4.0])
    print(numeric_gradient(function_2, x1))

    x2 = np.array([0.0, 2.0])
    print(numeric_gradient(function_2, x2))

    x3 = np.array([3.0, 0.0])
    print(numeric_gradient(function_2, x3))
