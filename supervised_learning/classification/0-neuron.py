#!/usr/bin/env python3

class Neuron:

    def __init__(self, nx):

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        
        self.W = 0
        self.b = 0
        self.A = 0
