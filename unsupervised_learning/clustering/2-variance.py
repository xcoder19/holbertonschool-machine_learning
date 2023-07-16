#!/usr/bin/env python3
"""variance"""

import numpy as np


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
