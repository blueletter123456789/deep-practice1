import numpy as np
import matplotlib.pyplot as plt


def numeric_diff(f, x):
    """Finite difference(using central difference)

    Args:
        f (func): function to be differentiated
        x (int|float): function arguments

    Returns:
        int: differential coefficient
    """
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    # sample function
    # y = 0.01x^2 + 0.1x
    return 0.01*x**2 + 0.1*x

def target_line(f, x):
    d = numeric_diff(f, x)
    # x=0の時のy
    y = f(x) - d*x
    return lambda t: d*t + y


if __name__ == '__main__':
    f = lambda x: x**2
    # analytic：2x→ 4.0
    print(numeric_diff(f, 2))

    f = lambda x: x**3
    # analytic：3x^2→ 48.0
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
