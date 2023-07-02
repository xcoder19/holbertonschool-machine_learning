#!/usr/bin/env python3
"""correlation"""
import numpy as np


def correlation(C):
    """correlation"""

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std_devs = np.sqrt(np.diag(C))

    corr_matrix = C / np.outer(std_devs, std_devs)

    return corr_matrix
