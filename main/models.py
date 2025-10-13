import logging
import os
import tensorflow as tf

logger = logging.getLogger("Models Module")

def create_autoencoder(input_dim, latent_dim=128):
    """
    Creates a standard Autoencoder model consisting of an Encoder and a Decoder.

    Args:
        input_dim (int): The number of input features for the omics view.
        latent_dim (int): The desired dimension of the latent (encoded) space.

    Returns:
        tuple: A tuple containing the full autoencoder model and the standalone encoder model.
    """
    # Encoder
    input_layer = tf.keras.layers.Input(shape=(input_dim,), name='encoder_input')
    encoded = tf.keras.layers.Dense(512, activation='relu')(input_layer)
    encoded = tf.keras.layers.Dense(0.3)(encoded)
    encoded = tf.keras.layers.Dense(256, activation='relu')(encoded)
    latent_space = tf.keras.layers.Dense(latent_dim, activation='relu', name='latent_space')(encoded)
    
    encoder = tf.keras.layers.Dense(inputs=input_layer, outputs=latent_space, name='encoder')

    # Decoder 
    decoder_input = tf.keras.layers.Input(shape=(latent_dim,), name='decoder_input')
    decoded = tf.keras.layers.Dense(256, activation='relu')(decoder_input)
    decoded = tf.keras.layers.Dropout(0.3)(decoded)
    decoded = tf.keras.layers.Dense(512, activation='relu')(decoded)
    reconstruction = tf.keras.layers.Dense(input_dim, activation='sigmoid', name='reconstruction')(decoded)
    
    decoder = tf.keras.layers.Model(inputs=decoder_input, outputs=reconstruction, name='decoder')

    # Full Autoencoder
    autoencoder_output = decoder(encoder(input_layer))
    autoencoder = tf.keras.layers.Model(inputs=input_layer, outputs=autoencoder_output, name='autoencoder')
    
    return autoencoder, encoder

def create_classifier(input_dim, num_classes):
    """
    Creates a standard classifier model to be trained on concatenated latent features.

    Args:
        input_dim (int): The dimension of the concatenated latent feature space.
        num_classes (int): The number of output classes for classification.

    Returns:
        tensorflow.keras.models.Model: The compiled Keras classifier model.
    """
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256, activation='relu')(input_layer)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    classifier = tf.keras.layers.Model(inputs=input_layer, outputs=output_layer)
    
    return classifier