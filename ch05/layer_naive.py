class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        """Forward in a multiple layer

        Args:
            x (float): Value of x
            y (float): Value of y

        Returns:
            float: Multiplication result
        """
        self.x = x
        self.y = y
        out = x * y

        return out
    
    def backward(self, dout):
        """Backward in a multiple layer

        Args:
            dout (float): Differentiation of the output of forward propagation

        Returns:
            float, float: each differential
        """
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        """Forwatd in a add layer

        Args:
            x (float): Value of x
            y (float): Value of y

        Returns:
            float: Result of addition
        """
        out = x + y
        return out
    
    def backward(self, dout):
        """Backward in a add layer

        Args:
            dout (float): Differentiation of the output of forwat propagation

        Returns:
            float, float: each differential
        """
        dx = dout * 1
        dy = dout * 1
        return dx, dy
