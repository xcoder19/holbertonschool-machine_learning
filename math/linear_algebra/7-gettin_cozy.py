#!/usr/bin/env python3
"""cat matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """cat matrices"""
    rows1 = len(mat1)
    rows2 = len(mat2)
    cols1 = len(mat1[0])
    cols2 = len(mat2[0])

    if axis == 0:
        if cols1 != cols2:
            return None
        matrix = []
        for x in mat1:
            matrix.append(x.copy())
        for x in mat2:
            matrix.append(x.copy())
        return matrix
    elif axis == 1:
        if rows1 != rows2:
            return None
        matrix = []
        for i in range(rows1):
            matrix.append(mat1[i] + mat2[i])
        return matrix
    else:
        return None
