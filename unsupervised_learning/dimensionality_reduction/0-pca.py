
#!/usr/bin/env python3
"""pca"""
import numpy as np


def pca(X, var=0.95):
    """pca"""

    _, singular_values, right_singular_vectors = np.linalg.svd(X)

    cumulative_variance = np.cumsum(singular_values)

    cumulative_variance_ratio = cumulative_variance / np.sum(singular_values)

    num_components = np.argwhere(cumulative_variance_ratio >= var)[0, 0]

    weights_matrix = right_singular_vectors[:num_components + 1].T

    return weights_matrix
