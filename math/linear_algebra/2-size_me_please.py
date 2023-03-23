#!/usr/bin/env python3
""" matrix shape"""
def matrix_shape(matrix):
    """ matrix shape"""
    rows = len(matrix)
    cols = 0
    cols2 = 0
    for i in range(0, rows):
        if isinstance(matrix[i], list) and cols < len(matrix[i]):
            cols = len(matrix[i])
            for j in matrix[i]:
                if isinstance(j, list) and cols2 < len(j):
                    cols2 = len(j)
    if cols2 > 0:
        shape = [rows, cols, cols2]
    elif cols > 0:
        shape = [rows, cols]
    else:
        shape = [rows]
    return shape
    
