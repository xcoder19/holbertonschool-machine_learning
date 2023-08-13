#!/usr/bin/env python3

"""GRUCell"""
import numpy as np


class GRUCell:
    """GRUCell"""

    def __init__(self, i, h, o):
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """sigmoid"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """forward"""
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        z = self.sigmoid(np.dot(concat_input, self.Wz) + self.bz)

        r = self.sigmoid(np.dot(concat_input, self.Wr) + self.br)

        concat_reset = np.concatenate((r * h_prev, x_t), axis=1)
        h_intermediate = np.tanh(np.dot(concat_reset, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_intermediate

        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1,
                               keepdims=True)

        return h_next, y
