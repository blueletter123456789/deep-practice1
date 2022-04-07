def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

if __name__ == '__main__':
    x = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for x1, x2 in x:
        print(AND(x1, x2))
