#!/usr/bin/env python3


determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """inverse"""
    if (not isinstance(matrix, list) or
        not all(isinstance(row, list) for row in matrix) or
            len(matrix) == 0):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adj = adjugate(matrix)

    inverse = [[element / det for element in row] for row in adj]

    return inverse
