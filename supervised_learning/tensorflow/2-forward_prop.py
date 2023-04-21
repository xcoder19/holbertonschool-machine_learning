#!/usr/bin/env python3
"""forward prop"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """forward prop"""
    prev_layer_output = x
    for i in range(len(layer_sizes)):
        prev_layer_output = create_layer(
            prev_layer_output, layer_sizes[i], activations[i])
    prediction = prev_layer_output
    return prediction
