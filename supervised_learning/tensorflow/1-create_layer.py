#!/usr/bin/env python3
"""layer method"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """layer method"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    layer = tf.layers.dense(

        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer')

    return layer
