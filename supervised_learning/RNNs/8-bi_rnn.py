#!/usr/bin/env python3


"""bi rnn"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """bi rnn"""
    t, m, i = X.shape
    h = h_0.shape[1]
    o = bi_cell.Wy.shape[1]

    H = np.zeros((t + 1, m, 2 * h))
    Y = np.zeros((t, m, o))

    H[0, :, :h] = h_0
    H[0, :, h:] = h_t

    for step in range(t):
        x_t = X[step]
        h_prev_f = H[step, :, :h]
        h_prev_b = H[step, :, h:]

        h_next_f = bi_cell.forward(h_prev_f, x_t)
        h_next_b = bi_cell.backward(h_prev_b, x_t)

        H[step + 1, :, :h] = h_next_f
        H[step + 1, :, h:] = h_next_b

        h_next_concat = np.concatenate((h_next_f, h_next_b), axis=1)
        Y[step] = bi_cell.output(h_next_concat)

    return H, Y
