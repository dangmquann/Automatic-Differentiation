import numpy as np

class Add:
    def forward(x, y):
        return np.add(x.value, y.value)
    def backward(g, x, y, z):
        return [g, g]
    
class Sub:
    def forward(x,y):
        return np.subtract(x.value, y.value)
    def backward(g, x, y, z):
        return [g, -g]
    
class Mul:
    def forward(x,y):
        return np.multiply(x.value, y.value)
    def backward(g, x, y, z):
        g_x = g * y.value
        g_y = g * x.value
        return [g_x, g_y]

class Div:
    def forward(x,y):
        return x.value / y.value
    def backward(g, x, y, z):
        g_x = g * (1 / y.value)
        g_y = - g * x.value / (y.value * y.value)
        return [g_x, g_y]
    