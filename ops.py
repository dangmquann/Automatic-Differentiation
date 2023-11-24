import numpy as np

class Add:
    def forward(x, y):
        return np.add(x.value, y.value)
    def backward(g, x, y, z):
        return [g, g]