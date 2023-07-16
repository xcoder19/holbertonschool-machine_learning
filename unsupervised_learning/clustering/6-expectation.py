#!/usr/bin/env python3
"""expectation"""
import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """expectation"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    try:
        n, d = X.shape
        k = pi.shape[0]

        g = np.zeros((k, n))
        for i in range(k):
            P = pdf(X, m[i], S[i])
            g[i] = pi[i] * P

        total = np.sum(g, axis=0)
        g /= total

        l = np.sum(np.log(total))
    except Exception:
        return None, None

    return g, l
