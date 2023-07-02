#!/usr/bin/env python3
"""multinormal"""
import numpy as np


class MultiNormal:
    """multinormal"""

    def __init__(self, data):
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        d, n = data.shape

        self.mean = np.mean(data, axis=1, keepdims=True)

        deviation = data - self.mean

        self.cov = deviation @ deviation.T / (n - 1)

    def pdf(self, x):
        """pdf"""

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        sqrt_det = np.sqrt(np.linalg.det(self.cov))
        const = 1 / (((2 * np.pi)**d * sqrt_det))

        diff = x - self.mean
        inv = np.linalg.inv(self.cov)
        exp = -0.5 * diff.T @ inv @ diff

        pdf = const * np.exp(exp)

        return pdf[0, 0]
