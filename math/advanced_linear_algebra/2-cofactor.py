#!/usr/bin/env python3
"""cofactor"""
determinant = __import__('0-determinant').determinant


def cofactor(matrix):
    """cofactor"""
    if (not isinstance(matrix, list) or
        not all(isinstance(row, list) for row in matrix) or
            len(matrix) == 0):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError('matrix must be a square matrix')

    if len(matrix) == 1:
        return [[1]]

    cofactors = []
    for r in range(len(matrix)):
        cofactorRow = []
        for c in range(len(matrix)):

            minor = [row[:c] + row[c + 1:]
                     for row in (matrix[:r] + matrix[r + 1:])]

            cofactorRow.append(((-1) ** (r + c)) * determinant(minor))
        cofactors.append(cofactorRow)
    return cofactors
