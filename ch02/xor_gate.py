from and_gate_bias import AND
from nand_gate_bias import NAND
from or_gate_bias import OR

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


if __name__ == '__main__':
    x = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for x1, x2 in x:
        print(XOR(x1, x2))
