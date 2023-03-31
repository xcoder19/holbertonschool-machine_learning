#!/usr/bin/env python3

"""poly_derivative"""


def poly_derivative(poly):
    """poly_derivative"""

    if not isinstance(
        poly,
        list) or any(
        not isinstance(
            coef,
            int) for coef in poly):
        return None

    if len(poly) <= 1:
        return None

    derivative = []
    for i in range(1, len(poly)):
        derivative.append(i * poly[i])

    return derivative
