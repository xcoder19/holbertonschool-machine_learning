#!/usr/bin/env python3

"""absorbing"""
import numpy as np


def absorbing(P):
    """absorbing"""
    if not isinstance(P, np.ndarray):
        return False

    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return False

    n = P.shape[0]

    for i in range(n):
        if np.allclose(P[i, i], 1.0) and np.allclose(
                P[i, :i], 0.0) and np.allclose(P[i, i + 1:], 0.0):
            return True

    return False
