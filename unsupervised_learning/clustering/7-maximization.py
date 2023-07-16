#!/usr/bin/env python3
"""maximization"""
import numpy as np


def maximization(X, g):
    """maximization"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    try:
        n, d = X.shape
        k, _ = g.shape

        pi = np.sum(g, axis=1) / n

        m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]

        S = np.zeros((k, d, d))
        for i in range(k):
            diff = X - m[i]
            S[i] = np.dot(g[i] * diff.T, diff) / np.sum(g, axis=1)[i]

    except Exception:
        return None, None, None

    return pi, m, S
