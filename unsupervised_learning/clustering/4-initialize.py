#!/usr/bin/env python3
"""Initialize"""
import numpy as np

kmeans = __import__("1-kmeans").kmeans


def initialize(X, k):
    """Initialize"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    try:
        pi = np.full(shape=(k,), fill_value=1 / k)

        m, _ = kmeans(X, k)

        d = X.shape[1]
        S = np.tile(np.identity(d), (k, 1, 1))
    except Exception:
        return None, None, None

    return pi, m, S
