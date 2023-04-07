#!/usr/bin/env python3
"""Exponential"""


class Exponential:
    """Exponential"""

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
            self.lambtha = 1 / float(sum(data) / len(data))

    def pdf(self, x):
        """returns pdf value """

        if x < 0:
            return 0

        return self.lambtha * (2.7182818285 ** (-self.lambtha * x))
