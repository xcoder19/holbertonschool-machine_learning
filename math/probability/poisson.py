#!/usr/bin/env python3
"""Poisson"""


class Poisson:
    """Poisson"""

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if not (lambtha > 0):
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """pmf"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        return ((2.7182818285**(-self.lambtha)) * (self.lambtha ** int(k))) / \
            self.factorial(int(k))

    def factorial(self, n):
        if n == 0 or n == 1:
            return 1
        else:
            return n * self.factorial(n - 1)
