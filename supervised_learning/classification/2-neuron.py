#!/usr/bin/env python3
"""Neuron class"""
import numpy as np


class Neuron:
    """Neuron class for supervised ml"""

    def __init__(self, nx):

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter for W"""
        return self.__W

    @property
    def b(self):
        """getter for b"""
        return self.__b

    @property
    def A(self):
        """getter for A"""
        return self.__A

    def forward_prop(self, X):
        """forward prop"""
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
