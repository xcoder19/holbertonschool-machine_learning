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

    @property
    def W1(self):
        """getter w1"""
        return self.__W1

    @property
    def b1(self):
        """getter b1"""
        return self.__b1

    @property
    def A1(self):
        """getter a1"""
        return self.__A1

    @property
    def W2(self):
        """getter w2"""
        return self.__W2

    @property
    def b2(self):
        """getter b2"""
        return self.__b2

    @property
    def A2(self):
        """getter a2"""
        return self.__A2

    def forward_prop(self, X):
        """forward prop"""
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """cost method"""
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) *
                                 np.log(1.0000001 - A)) + 0.0000000000000001

        return cost

    def evaluate(self, X, Y):
        """evaluate"""
        _, A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """gradien descent"""
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    import matplotlib.pyplot as plt

    def train(
            self,
            X,
            Y,
            iterations=5000,
            alpha=0.05,
            verbose=True,
            graph=True,
            step=100):
        """Train"""

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []

        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.A2)
            if verbose and i % step == 0:
                print(f"Cost after {i} iterations: {cost}")
                costs.append(cost)
                iters.append(i)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)

        if graph:
            import matplotlib.pyplot as plt
            plt.plot(iters, costs, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
