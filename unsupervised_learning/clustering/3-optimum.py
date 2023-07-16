#!/usr/bin/env python3
"""optimum"""
import numpy as np


kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """optimum"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if kmax is not None and (not isinstance(kmax, int)
                             or kmax <= 0 or kmax < kmin):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    results = []
    variances = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        var = variance(X, C)
        if var is None:
            return None, None
        results.append((C, clss))
        variances.append(var)

    d_vars = [variances[i] - variances[i - 1] if i > 0 else variances[i]
              for i in range(len(variances))]

    return results, d_vars
