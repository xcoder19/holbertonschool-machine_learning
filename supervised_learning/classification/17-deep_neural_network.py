#!/usr/bin/env python3

"""deep neural network"""
import numpy as np


class DeepNeuralNetwork:
    """deep neural network"""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError("layers must be a list of positive integers")

        for i in range(1, self.__L + 1):
            he_et_al = np.sqrt(2 / layer_size)
            self.__weights["W" + str(i)] = np.random.randn(
                layers[i - 1], layer_size) * he_et_al
            self.__weights["b" + str(i)] = np.zeros((layers[i - 1], 1))
            layer_size = layers[i - 1]

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
