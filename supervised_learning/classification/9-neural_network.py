#!/usr/bin/env python3
"""neural network"""
import numpy as np


class NeuralNetwork:
    """neural network"""

    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    def get_W1(self):
        """getter w1"""
        return self.__W1

    def get_b1(self):
        """getter b1"""
        return self.__b1

    def get_A1(self):
        """getter a1"""
        return self.__A1

    def get_W2(self):
        """getter w2"""
        return self.__W2

    def get_b2(self):
        """getter b2"""
        return self.__b2

    def get_A2(self):
        """getter a2"""
        return self.__A2
