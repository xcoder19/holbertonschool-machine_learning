#!/usr/bin/env python3
def matrix_transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    result = [[matrix[j][i] for j in range(rows)] for i in range(cols)]

    return result
