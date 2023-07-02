#!/usr/bin/env python3
"""definiteness"""
import numpy as np


def definiteness(matrix):
    """definiteness"""

    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.conj().T):
        return None

    eigenvalues = np.linalg.eigvalsh(matrix)

    if np.all(eigenvalues > 0):
        return 'Positive definite'
    elif np.all(eigenvalues >= 0):
        return 'Positive semi-definite'
    elif np.all(eigenvalues < 0):
        return 'Negative definite'
    elif np.all(eigenvalues <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
