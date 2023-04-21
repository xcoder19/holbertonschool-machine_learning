#!/usr/bin/env python3
""" placeholders"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """placeholder method"""
    x = tf.placeholder(dtype=tf.float32, shape=(nx, None), name='x')
    y = tf.placeholder(dtype=tf.float32, shape=(classes, None), name='y')
    return x, y
