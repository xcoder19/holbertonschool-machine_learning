#!/usr/bin/env python3

"""adjugate"""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """adjugate"""

    cofactor_matrix = cofactor(matrix)

    adjugate_matrix = list(map(list, zip(*cofactor_matrix)))

    return adjugate_matrix
