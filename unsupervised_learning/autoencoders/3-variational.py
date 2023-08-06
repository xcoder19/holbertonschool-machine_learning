#!/usr/bin/env python3
"""variational autoencoder"""
import tensorflow.keras as keras


def sampling(args):
    """sampling"""
    mean, log_var = args
    epsilon = keras.backend.random_normal(
        shape=keras.backend.shape(mean), mean=0.0, stddev=1.0)
    return mean + keras.backend.exp(0.5 * log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """variational autoencoder"""

    encoder_input = keras.layers.Input(shape=(input_dims,))
    encoder_layers = encoder_input

    for nodes in hidden_layers:
        encoder_layers = keras.layers.Dense(
            nodes, activation='relu')(encoder_layers)

    latent_mean = keras.layers.Dense(
        latent_dims, activation=None)(encoder_layers)
    latent_log_var = keras.layers.Dense(
        latent_dims, activation=None)(encoder_layers)

    latent_sample = keras.layers.Lambda(
        sampling)([latent_mean, latent_log_var])

    encoder = keras.models.Model(
        encoder_input, [
            latent_sample, latent_mean, latent_log_var], name='encoder')

    decoder_input = keras.layers.Input(shape=(latent_dims,))
    decoder_layers = decoder_input

    for nodes in reversed(hidden_layers):
        decoder_layers = keras.layers.Dense(
            nodes, activation='relu')(decoder_layers)

    decoder_output = keras.layers.Dense(
        input_dims, activation='sigmoid')(decoder_layers)
    decoder = keras.models.Model(decoder_input, decoder_output,
                                 name='decoder')

    autoencoder_input = keras.layers.Input(shape=(input_dims,))
    encoded_sample, encoded_mean, \
        encoded_log_var = encoder(autoencoder_input)
    decoded = decoder(encoded_sample)
    autoencoder_model = keras.models.Model(
        autoencoder_input, decoded, name='autoencoder')

    autoencoder_model.compile(optimizer='adam',
                              loss='binary_crossentropy')

    return encoder, decoder, autoencoder_model
