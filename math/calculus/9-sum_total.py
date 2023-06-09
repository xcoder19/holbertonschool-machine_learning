#!/usr/bin/env python3
"""summation_i_squared"""


def summation_i_squared(n):
    """summation_i_squared"""

    if not isinstance(n, int) or n < 1:
        return None

    return (n * (n + 1) * (2 * n + 1)) // 6
