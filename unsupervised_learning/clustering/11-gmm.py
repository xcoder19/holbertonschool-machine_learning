#!/usr/bin/env python3
"""GMM"""
from sklearn.mixture import GaussianMixture
import numpy as np


def gmm(X, k):
    """GMM"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None, None, None, None

    gmm = GaussianMixture(n_components=k).fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)

    return pi, m, S, clss, bic
