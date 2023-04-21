#!/usr/bin/env python3

"""train op"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """train op"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op
