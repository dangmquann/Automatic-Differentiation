from utils import check
from ops import *

from inspect import signature
import numpy as np


class Tensor:
    def __init__(self, value, _children = [], requires_grad = True):
        self.value = np.atleast_2d(value)
        self.requires_grad = requires_grad
        self.grad = np.zeros(np.shape(self.value))

        self._op = "leaf"
        
        self._forward = lambda: self.value
        self._outgrad = lambda x, y: () 
        self._children = _children


    ### --- Operator Overloading --- ###

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        return OP(Add, self, check(other, Tensor))

    def __radd__(self, other):
        return OP(Add, check(other, Tensor), self)

    def __sub__(self, other):
        return OP(Sub, self, check(other, Tensor))

    def __rsub__(self, other): 
        return OP(Sub, check(other, Tensor), self)

    def __pow__(self, other):
        return OP(Pow, self, check(other, Tensor))

    def __mul__(self, other):
        return OP(Mul, self, check(other, Tensor))

    def __rmul__(self, other):
        return OP(Mul, check(other, Tensor), self)

    def __div__(self, other):
        return OP(Div, self, check(other, Tensor))

    def __rdiv__(self, other):
        return OP(Div, check(other, Tensor), self)

    def __truediv__(self, other):
        return OP(Div, self, check(other, Tensor))

    def __rtruediv__(self, other):
        return OP(Div, check(other, Tensor), self)


    ### --- Properties --- ###

    @property
    def shape(self):
        return self.value.shape

    
    ### --- Backprop & Computation Graph Functions --- ###

    def topo_sort(self):
        topo = []
        visited = set()

        def recurse(tensor):
            if tensor not in visited:
                visited.add(tensor)
    
                for child in tensor._children:
                   recurse(child)
    
                topo.insert(0, tensor)
    
        recurse(self)

        self._topo = topo

    def backward(self):
        self.topo_sort()

        self.grad = np.ones(np.shape(self.value))

        for tensor in self._topo:
            # This is a GOTCHA of the current implementation, even if
            # requires_grad = True, the gradient still gets computed. 
            grad = tensor._outgrad(tensor.grad, *tensor._children, tensor)

            for child, ingrad in zip(tensor._children, grad):
                if child.requires_grad:
                    child.grad = child.grad + ingrad


### ----- OP BUILDER ----- ### 

def OP(op, *args, **kwargs):
    value = op.forward(*args, **kwargs)

    tensors = [arg for arg in args if isinstance(arg, Tensor)]   #arg: tensor 

    requires_grad = True if np.any([tensor.requires_grad for tensor in tensors]) else False

    output_tensor = Tensor(value, tensors, requires_grad)

    output_tensor._outgrad = op.backward
    output_tensor._forward = op.forward
    output_tensor._op = op.__name__.lower()

    return output_tensor


