#!/usr/bin/env python3

import numpy as np
"""operations on matrices with numpy"""


def np_elementwise(mat1, mat2):
    """operations on matrices with numpy"""
    sum = mat1 + mat2
    diff = mat1 - mat2
    product = mat1 * mat2
    div = mat1 / mat2
    return (sum, diff, product, div)
