#!/usr/bin/env python3
""" add arrays"""


def add_arrays(arr1, arr2):
    """add arrays"""
    arr = []
    if len(arr1) != len(arr2):
        return None
    for i in range(len(arr1)):
        arr.append(arr1[i] + arr2[i])
    return arr
