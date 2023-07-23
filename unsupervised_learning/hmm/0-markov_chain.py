#!/usr/bin/env python3

"""markov chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """markov chain"""
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None

    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None

    if len(s.shape) != 2 or s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None

    if not isinstance(t, int) or t < 1:
        return None

    n = P.shape[0]
    for _ in range(t):
        s = np.dot(s, P)

    return s
