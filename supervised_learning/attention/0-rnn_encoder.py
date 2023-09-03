#!/usr/bin/env python3
"""RNNEncoder"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNNEncoder"""

    def __init__(self, vocab, embedding, units, batch):
        """init"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """initialize_hidden_state"""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """call"""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
