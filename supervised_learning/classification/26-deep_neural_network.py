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

    def evaluate(self, X, Y):
        """
        evaluates  neural network
        """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        gradient descent
        """
        m = Y.shape[1]
        dz = cache['A' + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            dw = (1 / m) * np.dot(dz, A_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            dz = np.dot(self.__weights['W' + str(i)].T,
                        dz) * (A_prev * (1 - A_prev))
            self.__weights['W' + str(i)] -= alpha * dw
            self.__weights['b' + str(i)] -= alpha * db

    import matplotlib.pyplot as plt

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the deep neural network
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
            
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                costs.append(cost)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))

        if graph is True:
            import matplotlib.pyplot as plt
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()
        
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        saves to a file in pickle format
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            import pickle
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        loads a pickled DeepNeuralNetwork object
        """
        import os
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as file:
            import pickle
            return pickle.load(file)   