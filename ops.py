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
    
class Sin:
    def forward(x):
        return np.sin(x.value)
    
    def backward(g, x, y):
        return [g * np.cos(x.value)]

class Cos:
    def forward(x):
        return np.cos(x.value)
    
    def backward(g, x, y):
        return [-g * np.sin(x.value)]

class Tan:
    def forward(x):
        return np.tan(x.value)
    
    def backward(g, x, y):
        return [g * (1 / np.cos(x.value))**2]