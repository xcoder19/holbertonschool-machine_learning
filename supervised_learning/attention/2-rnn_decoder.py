#!/usr/bin/env python3


"""RNNDecoder"""
import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNNDecoder"""

    def __init__(self, vocab, embedding, units, batch):
        """init"""
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """call method"""
        x = self.embedding(x)
        context, _ = SelfAttention(units=self.units)(s_prev, hidden_states)
        context = tf.expand_dims(context, axis=1)
        x = tf.concat([context, x], axis=-1)
        y, s = self.gru(x, initial_state=s_prev)
        y = self.F(y)
        return y, s
