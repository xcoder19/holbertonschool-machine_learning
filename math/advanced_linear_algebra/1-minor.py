#!/usr/bin/env python3
"""minor"""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """minor"""
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            submatrix = [row[:j] + row[j + 1:]
                         for row in (matrix[:i] + matrix[i + 1:])]
            if len(submatrix) == 0:
                minor_row.append(1)
            else:
                minor_row.append(determinant(submatrix))
        minor_matrix.append(minor_row)

    return minor_matrix
