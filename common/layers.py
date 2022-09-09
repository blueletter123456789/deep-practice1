class Relu:
    def __init__(self):
        """initialize RELU class

        mask (ndarray[bool])
        """
        self.mask = None
    
    def forward(self, x):
        """Forward in a RELU layer

        Args:
            x (ndarray): Value of x

        Returns:
            ndarray: Result of RELU
        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out
    
    def backward(self, dout):
        """Backward in a RELU layer

        Args:
            dout (ndarray): Differentiation of ndarray type

        Returns:
            ndarray: Differentiation of ndarray type
        """
        dout[self.mask] = 0

        dx = dout

        return dx
