#!/usr/bin/env python3
"""determinant"""


def determinant(matrix):
    """determinant"""
    if not all(isinstance(i, list) for i in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix != [[]]:
        rows = len(matrix)
        for i in range(rows):
            cols = len(matrix[i])
            if cols != rows:
                raise ValueError("matrix must be a square matrix")
        if rows == 1:
            return matrix[0][0]
        if (rows == 2):
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        elif rows >= 3:
            det = 0
            for c in range(rows):
                submatrix = [row[:c] + row[c + 1:]
                             for row in matrix[1:]]
                sign = (-1) ** c
                sub_det = determinant(submatrix)
                det += sign * matrix[0][c] * sub_det
            return det
    else:
        return 1
