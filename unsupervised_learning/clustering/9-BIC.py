#!/usr/bin/env python3
"""BIC"""

import numpy as np

expectation_maximization = __import__("8-EM").expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """BIC"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    ks = range(kmin, kmax + 1)
    l = np.empty(len(ks))
    b = np.empty(len(ks))
    best_b = np.inf
    best_k = None
    best_result = None

    for i, k in enumerate(ks):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)

        if pi is None or m is None or S is None:
            return None, None, None, None

        p = k * d * (d + 1) / 2 + d * k + k - 1  # Number of parameters
        BIC = p * np.log(n) - 2 * log_likelihood

        l[i] = log_likelihood
        b[i] = BIC

        if BIC < best_b:
            best_b = BIC
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, l, b
