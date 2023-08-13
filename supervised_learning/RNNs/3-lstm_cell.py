#!/usr/bin/env python3
"""LSTMCell"""
import numpy as np


class LSTMCell:
    """LSTMCell"""

    def __init__(self, i, h, o):
        """init"""
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """sigmoid"""
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """tanh function"""
        return np.tanh(x)

    def forward(self, h_prev, c_prev, x_t):
        """forward"""
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        f = self.sigmoid(np.dot(concat_input, self.Wf) + self.bf)

        u = self.sigmoid(np.dot(concat_input, self.Wu) + self.bu)

        c_intermediate = u * self.tanh(np.dot(concat_input, self.Wc) + self.bc)

        c_next = f * c_prev + c_intermediate

        o = self.sigmoid(np.dot(concat_input, self.Wo) + self.bo)

        h_next = o * self.tanh(c_next)

        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1,
                               keepdims=True)

        return h_next, c_next, y
