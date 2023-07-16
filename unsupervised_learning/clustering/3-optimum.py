#!/usr/bin/env python3
"""optimum"""
import numpy as np


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


def variance(X, C):
    """variance"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    try:
        distances = np.linalg.norm(X[:, None] - C, axis=-1)

        min_distances = np.min(distances, axis=-1)

        var = np.sum(min_distances ** 2)
    except Exception:
        return None

    return var


def kmeans(X, k, iterations=1000):
    """kmeans"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = initialize(X, k)

    for _ in range(iterations):
        C_old = C.copy()

        distances = np.linalg.norm(X[:, None] - C, axis=-1)

        clss = np.argmin(distances, axis=-1)

        for j in range(k):
            if X[clss == j].size == 0:
                C[j] = initialize(X, 1)
            else:
                C[j] = np.mean(X[clss == j], axis=0)

        if np.all(C_old == C):
            break

    return C, clss


def initialize(X, k):
    """Initialize """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    centroids = np.random.uniform(min_vals, max_vals, (k, d))

    return centroids
