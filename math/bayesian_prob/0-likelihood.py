#!/usr/bin/env python3
"""likelihood"""
import numpy as np


def likelihood(x, n, P):
    """likelihood"""

    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not np.all((0 <= P) & (P <= 1)):
        raise ValueError('All values in P must be in the range [0, 1]')

    binom_coef = np.math.factorial(
        n) / (np.math.factorial(x) * np.math.factorial(n - x))

    likelihood = binom_coef * (P**x) * ((1 - P)**(n - x))

    return likelihood
