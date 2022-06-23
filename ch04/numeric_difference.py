import numpy as np
import matplotlib.pyplot as plt


def numeric_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x

def target_line(f, x):
    d = numeric_diff(f, x)
    # x=0の時のy
    y = f(x) - d*x
    return lambda t: d*t + y


if __name__ == '__main__':
    # 解析的：2x
    f = lambda x: x**2
    print(numeric_diff(f, 2))

    # 解析的：3x^2
    f = lambda x: x**3
    print(numeric_diff(f, 4))


    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")

    tf = target_line(function_1, 5)
    y2 = tf(x)

    tf2 = target_line(function_1, 10)
    y3 = tf2(x)

    plt.plot(x, y)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.show()
