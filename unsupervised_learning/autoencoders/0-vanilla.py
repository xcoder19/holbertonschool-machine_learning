#!/usr/bin/env python3
"""vanila autoencoder"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def autoencoder(input_dims, hidden_layers, latent_dims):
    """vanila autoencoder"""

    encoder_input = Input(shape=(input_dims,))
    encoder_layers = []
    for nodes in hidden_layers:
        encoder_layers.append(Dense(nodes, activation='relu')(encoder_input))
    encoder_output = Dense(latent_dims, activation='relu')(encoder_layers[-1])
    encoder = Model(encoder_input, encoder_output, name='encoder')

    decoder_input = Input(shape=(latent_dims,))
    decoder_layers = []
    for nodes in reversed(hidden_layers):
        decoder_layers.append(Dense(nodes, activation='relu')(decoder_input))
    decoder_output = Dense(
        input_dims, activation='sigmoid')(decoder_layers[-1])
    decoder = Model(decoder_input, decoder_output, name='decoder')

    autoencoder_input = Input(shape=(input_dims,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder_model = Model(autoencoder_input, decoded, name='autoencoder')

    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder_model
