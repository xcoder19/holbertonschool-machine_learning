#!/usr/bin/env python3
"""pca"""

import numpy as np


def pca(X, ndim):
    """
    pca
    """
    X_normalized = X - np.mean(X, axis=0)

    _, _, right_singular_vectors = np.linalg.svd(X_normalized)

    transformation_matrix = right_singular_vectors.T[:, :ndim]

    X_transformed = np.dot(X_normalized, transformation_matrix)

    return X_transformed
