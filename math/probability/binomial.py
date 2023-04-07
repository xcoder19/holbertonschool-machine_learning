#!/usr/bin/env python3
"""Binomial"""


class Binomial:
    """Binomial"""

    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if not (isinstance(n, int) and n > 0):
                raise ValueError("n must be a positive value")
            if not (isinstance(p, float) and 0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.n, self.p = self.helper(data)

    def helper(self, data):
        """returns values of  n and p """

        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        p = 1 - variance / mean
        n = round(mean / p)
        p = mean / n
        return n, p
