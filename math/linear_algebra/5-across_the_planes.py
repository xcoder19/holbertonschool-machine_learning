#!/usr/bin/env python3
"""add_matrice2d"""


def add_matrices2D(mat1, mat2):
    """add_matrice2d"""
    matrix = []
    rows = len(mat1)
    cols = len(mat1[0])
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    for i in range(rows):
        arr = []
    for j in range(cols):
        arr.append(mat1[i][j] + mat2[i][j])
    matrix.append(arr)

    return matrix


def matrix_shape(matrix):
    """ matrix shape"""
    shape = []
    if len(matrix) == 0:
        return None
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
