#!/usr/bin/env python3
"""conv autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """conv autoencoder"""
    encoder_input = keras.layers.Input(shape=input_dims)
    encoder_layers = encoder_input

    for num_filters in filters:
        encoder_layers = keras.layers.Conv2D(
            num_filters, (3, 3), activation='relu',
            padding='same')(encoder_layers)
        encoder_layers = keras.layers.MaxPooling2D(
            (2, 2), padding='same')(encoder_layers)

    encoder_output = encoder_layers
    encoder = keras.models.Model(encoder_input, encoder_output,
                                 name='encoder')

    decoder_input = keras.layers.Input(shape=latent_dims)
    decoder_layers = decoder_input

    for num_filters in reversed(filters[:-1]):
        decoder_layers = \
            keras.layers.Conv2D(
                num_filters, (3, 3),
                activation='relu',
                padding='same')(decoder_layers)
        decoder_layers = keras.layers.UpSampling2D((2,
                                                    2))(decoder_layers)

    decoder_layers = keras.layers.Conv2D(
        filters[-1], (3, 3), activation='relu',
        padding='valid')(decoder_layers)
    decoder_layers = keras.layers.UpSampling2D((2,
                                                2))(decoder_layers)

    decoder_output = keras.layers.Conv2D(
        input_dims[-1], (3, 3), activation='sigmoid',
        padding='same')(decoder_layers)

    decoder = keras.models.Model(decoder_input,
                                 decoder_output, name='decoder')

    autoencoder_input = keras.layers.Input(shape=input_dims)
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder_model = keras.models.Model(
        autoencoder_input, decoded, name='autoencoder')

    autoencoder_model.compile(optimizer='adam',
                              loss='binary_crossentropy')

    return encoder, decoder, autoencoder_model
