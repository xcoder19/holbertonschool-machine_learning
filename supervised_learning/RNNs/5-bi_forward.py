#!/usr/bin/env python3

"""BidirectionalCell"""
import numpy as np


class BidirectionalCell:

    """BidirectionalCell"""

    def __init__(self, i, h, o):
        """init"""
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h * 2, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward"""
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        h_next_f = np.tanh(np.dot(concat_input, self.Whf) + self.bhf)
        h_next_b = np.tanh(np.dot(concat_input, self.Whb) + self.bhb)

        h_next = np.concatenate((h_next_f, h_next_b), axis=1)

        return h_next
