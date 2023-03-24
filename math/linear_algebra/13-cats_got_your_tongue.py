#!/usr/bin/env python3

import numpy as np
"""concatenate matrices using numpy"""


def np_cat(mat1, mat2, axis=0):
    """concatenate matrices using numpy"""
    return np.concatenate((mat1, mat2), axis=axis)
