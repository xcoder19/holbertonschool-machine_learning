#!/usr/bin/env python3
"""one hot decoder"""
import numpy as np


def one_hot_decode(one_hot):
    """
    one_hot_decode
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
