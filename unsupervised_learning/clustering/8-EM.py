#!/usr/bin/env python3
"""expectation_maximization"""
import numpy as np


initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """expectation_maximization"""
    # Check inputs
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    try:
        pi, m, S = initialize(X, k)
        old_l = None

        for i in range(iterations):
            g, l = expectation(X, pi, m, S)
            pi, m, S = maximization(X, g)

            if verbose and (i % 10 == 0):
                print(
                    "Log Likelihood after {} iterations: {}".format(
                        i, round(
                            l, 5)))

            if old_l is not None and abs(old_l - l) <= tol:
                break

            old_l = l

        if verbose:
            print(
                "Log Likelihood after {} iterations: {}".format(
                    i, round(
                        l, 5)))
    except Exception:
        return None, None, None, None, None

    return pi, m, S, g, l
