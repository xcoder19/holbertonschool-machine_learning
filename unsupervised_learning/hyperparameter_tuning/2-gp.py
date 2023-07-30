#!/usr/bin/env python3
"""GaussianProcess"""
import numpy as np


class GaussianProcess:
    """GaussianProcess"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """kernel"""
        m = X1.shape[0]
        n = X2.shape[0]
        K = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                dist = np.linalg.norm(X1[i] - X2[j])
                K[i, j] = self.sigma_f**2 * np.exp(-0.5 * (dist / self.l)**2)

        return K

    def predict(self, X_s):
        """predict"""
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        K_s = self.kernel(self.X, X_s)
        mu_s = K_s.T.dot(K_inv).dot(self.Y).flatten()

        sigma_s = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))
        return mu_s, sigma_s

    def update(self, X_new, Y_new):
        """update"""
        self.X = np.append(self.X, X_new)
        self.Y = np.append(self.Y, Y_new)
        self.K = self.kernel(self.X, self.X)
        self.Y = np.array([self.Y])
        self.Y = self.Y.reshape(-1, 1)
        self.X = np.array([self.X])
        self.X = self.X.reshape(-1, 1)
