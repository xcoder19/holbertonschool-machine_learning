#!/usr/bin/env python3
"""normalization_constants"""
import numpy as np


def normalization_constants(X):
    """normalization_constants"""

    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)

    return mean, std_dev
