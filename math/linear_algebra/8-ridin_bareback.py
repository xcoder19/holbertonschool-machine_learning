#!/usr/bin/env python3
"""matrix multiplication"""


def mat_mul(mat1, mat2):
    """matrix multiplication"""
    rows1 = len(mat1)
    rows2 = len(mat2)
    cols1 = len(mat1[0])
    cols2 = len(mat2[0])

    if cols1 != rows2:
        return None

    matrix = [[0 for j in range(cols2)] for i in range(rows1)]

    for i in range(rows1):
        for j in range(cols2):
            for k in range(rows2):
                matrix[i][j] += mat1[i][k] * mat2[k][j]

    return matrix
