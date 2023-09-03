#!/usr/bin/env python3
"""positional_encoding"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """positional_encoding"""
    positional_encoding = np.zeros((max_seq_len, dm))
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            angle = pos / np.power(10000, (2 * i) / np.float32(dm))
            positional_encoding[pos, i] = np.sin(angle)
            positional_encoding[pos, i + 1] = np.cos(angle)
    return positional_encoding
