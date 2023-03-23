#!/usr/bin/env python3
def matrix_shape(matrix):
    rows = len(matrix)
    cols = 0
    cols2 = 0
    for i in range(0,rows):
        if cols < len(matrix[i]):
            cols = len(matrix[i])
        for j in matrix[i]:
            if isinstance(j,list) and cols2 < len(j):
                cols2 = len(j)
    if cols2 > 0:
        shape = [rows,cols,cols2]
    else:
        shape = [rows,cols]
    return shape
    
