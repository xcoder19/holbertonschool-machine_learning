#!/usr/bin/env python3
"""calculatesthe probability density function of a Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """calculatesthe probability density function of a Gaussian distribution"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    try:
        n, d = X.shape

        S_inv = np.linalg.inv(S)
        det = np.linalg.det(S)

        diff = X - m
        exponent = -0.5 * np.einsum('ij,jk,ik->i', diff, S_inv, diff)

        norm_const = 1.0 / (np.sqrt((2 * np.pi) ** d * det))

        P = norm_const * np.exp(exponent)

        P = np.maximum(P, 1e-300)
    except Exception:
        return None

    return P
