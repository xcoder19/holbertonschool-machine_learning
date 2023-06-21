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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if (not isinstance(layers[i], int)) or (layers[i] < 1):
                raise TypeError("layers must be a list of positive integers")

            key = "W" + str(i + 1)
            self.__weights[key] = np.random.randn(
                layers[i], nx) * np.sqrt(2 / nx)

            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))
            nx = layers[i]

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        forward propagation of the neural network
        """
        self.__cache['A0'] = X
        for i in range(self.__L):
            W = self.__weights['W' + str(i + 1)]
            A = self.__cache['A' + str(i)]
            b = self.__weights['b' + str(i + 1)]
            Z = np.dot(W, A) + b

            A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(i + 1)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """ calculates the cost """
        m = Y.shape[1]
        term1 = Y * np.log(A)
        term2 = (1 - Y) * np.log(1.0000001 - A)
        cost = -(1 / m) * np.sum(term1 + term2)
        return cost
