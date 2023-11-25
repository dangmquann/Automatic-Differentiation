import numpy as np


def check(x, Type): 
    return x if isinstance(x, Type) else Type(x)

