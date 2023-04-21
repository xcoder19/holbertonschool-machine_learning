#!/usr/bin/env python3

class Neuron:

    def __init__(self, nx):

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        import numpy as np
        self.W = np.random.randn(nx).reshape(1, nx)
        self.b = 0
        self.A = 0
