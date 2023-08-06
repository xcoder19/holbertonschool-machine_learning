#!/usr/bin/env python3
"""vanila autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """vanila autoencoder"""
    encoder_input = keras.layers.Input(shape=(input_dims,))
    encoder_layers = []
    for nodes in hidden_layers:
        encoder_layers.append(keras.layers.Dense(
            nodes, activation='relu')(encoder_input))
    encoder_output = keras.layers.Dense(
        latent_dims, activation='relu')(encoder_layers[-1])
    encoder = keras.models.Model(encoder_input, encoder_output, name='encoder')

    decoder_input = keras.layers.Input(shape=(latent_dims,))
    decoder_layers = []
    for nodes in reversed(hidden_layers):
        decoder_layers.append(keras.layers.Dense(
            nodes, activation='relu')(decoder_input))
    decoder_output = keras.layers.Dense(
        input_dims, activation='sigmoid')(decoder_layers[-1])
    decoder = keras.models.Model(decoder_input, decoder_output, name='decoder')

    autoencoder_input = keras.layers.Input(shape=(input_dims,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder_model = keras.models.Model(
        autoencoder_input, decoded, name='autoencoder')

    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder_model
