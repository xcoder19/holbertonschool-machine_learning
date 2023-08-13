#!/usr/bin/env python3

"""deep rnn"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """deep rnn"""
    t, m, i = X.shape
    l_ = len(rnn_cells)
    h = h_0.shape[2]
    o = rnn_cells[-1].Wy.shape[1]

    H = np.zeros((t + 1, l_, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        x_t = X[step]
        h_prev_layer = H[step]

        h_next_layer = [h_prev_layer[0]]

        for layer in range(1, l_):
            rnn_cell = rnn_cells[layer]
            h_prev = h_next_layer[layer - 1]
            h_next, _ = rnn_cell.forward(h_prev, h_prev_layer[layer])
            h_next_layer.append(h_next)

        H[step + 1] = np.array(h_next_layer)

        Y[step] = h_next_layer[-1]

    return H, Y
