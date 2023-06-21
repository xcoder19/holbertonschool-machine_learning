#!/usr/bin/env python3
""" one hot encoder"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    one_hot_encode
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None
