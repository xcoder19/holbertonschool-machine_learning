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

        self._W = np.random.randn(nx).reshape(1, nx)
        self._b = 0
        self._A = 0
