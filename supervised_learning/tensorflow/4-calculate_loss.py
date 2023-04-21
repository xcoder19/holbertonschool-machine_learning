#!/usr/bin/env python3

"""calculate loss"""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """calculate loss"""
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=y_pred))
    return loss
