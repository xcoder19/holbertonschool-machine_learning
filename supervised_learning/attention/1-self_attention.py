
#!/usr/bin/env python3

"""SelfAttention"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """SelfAttention"""

    def __init__(self, units):
        """init"""
        super(SelfAttention, self).__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """call method"""
        s_prev = tf.expand_dims(s_prev, axis=1)
        alignment = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))

        attention_weights = tf.nn.softmax(alignment, axis=1)

        context = tf.reduce_sum(attention_weights * hidden_states, axis=1)

        return context, attention_weights
