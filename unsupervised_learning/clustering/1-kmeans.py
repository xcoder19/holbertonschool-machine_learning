#!/usr/bin/env python3
"""kmeans"""
import numpy as np


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
