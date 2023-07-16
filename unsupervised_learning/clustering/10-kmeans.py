#!/usr/bin/env python3
"""kmeans"""

from sklearn.cluster import KMeans


def kmeans(X, k):
    """kmeans"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None

    kmeans = KMeans(n_clusters=k).fit(X)

    C = kmeans.cluster_centers_

    clss = kmeans.labels_

    return C, clss
