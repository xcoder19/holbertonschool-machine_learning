#!/usr/bin/env python3
"""regular"""


import numpy as np


def regular(P):
    """regular"""
    if not isinstance(P, np.ndarray):
        return None

    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None

    if not np.allclose(P.sum(axis=1), 1.0):
        return None

    n = P.shape[0]

    if np.allclose(P, np.eye(n)):
        return None

    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    index = np.argmin(np.abs(eigenvalues - 1))

    steady_state_probabilities = np.real_if_close(
        np.abs(eigenvectors[:, index].T) / np.sum(np.abs(eigenvectors[:, index])))

    steady_state_probabilities_np = np.array([steady_state_probabilities])
    return steady_state_probabilities_np
