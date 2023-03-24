#!/usr/bin/env python3
"""concatenate matrices using numpy"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenate matrices using numpy"""
    return np.concatenate((mat1, mat2), axis=axis)
