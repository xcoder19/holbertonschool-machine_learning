#!/usr/bin/env python3
"""cat matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """cat matrices"""
    matrix = []
    if axis == 0:
        for x in mat1:
            matrix.append(x)

        for y in mat2:
            matrix.append(y)
    elif axis == 1:
        for i in range(len(mat1)):
            matrix.append(cat_arrays(mat1[i], mat2[i]))

    return matrix


def cat_arrays(arr1, arr2):
    """concat arrays"""
    arr = []

    for x in arr1:
        arr.append(x)

    for y in arr2:
        arr.append(y)

    return arr
