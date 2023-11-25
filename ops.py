#from utils import broadcast
import numpy as np


#@broadcast
class Add:
    def forward(x, y):
        return np.add(x.value, y.value)

    def backward(g, x, y, z):
        return [g, g]

#@broadcast
class Sub:
    def forward(x, y):
        return x.value - y.value

    def backward(g, x, y, z):
        return [g, -g]

#@broadcast
class Mul:
    def forward(x, y):
        return x.value * y.value

    def backward(g, x, y, z):
        g_x = g * y.value
        g_y = g * x.value

        return [g_x, g_y]

#@broadcast
class Div:
    def forward(x, y):
        return x.value / y.value

    def backward(g, x, y, z):
        g_x = g * (1.0 / y.value)
        g_y = g * (-x.value / (y.value ** 2))

        return [g_x, g_y]

#@broadcast
class Pow:
    def forward(x, y):
        return x.value ** y.value

    def backward(g, x, y, z):
        # Note: We do not compute the 'y' grad.
        g_x = g * y.value * np.power(x.value, y.value - 1)
        g_y = np.array([[0]])

        return [g_x, g_y]
    