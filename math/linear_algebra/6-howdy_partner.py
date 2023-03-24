#!/usr/bin/env python3
"""concat arrays"""


def cat_arrays(arr1, arr2):
    """concat arrays"""
    arr = []

    for x in arr1:
        arr.append(x)

    for y in arr2:
        arr.append(y)

    return arr
