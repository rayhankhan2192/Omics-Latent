import tensorflow as tf

def create_autoencoder(input_dim, latent_dim=128):
    """
    Creates a standard Autoencoder model consisting of an Encoder and a Decoder.

    Returns:
        tuple: A tuple containing:
            (autoencoder_model, encoder_model, decoder_model)
    """
    # Encoder
    input_layer = tf.keras.layers.Input(shape=(input_dim,), name='encoder_input')
    encoded = tf.keras.layers.Dense(512, activation='relu')(input_layer)
    encoded = tf.keras.layers.Dropout(0.3)(encoded)
    encoded = tf.keras.layers.Dense(256, activation='relu')(encoded)
    latent_space = tf.keras.layers.Dense(latent_dim, activation='relu', name='latent_space')(encoded)
    
    encoder = tf.keras.models.Model(inputs=input_layer, outputs=latent_space, name='encoder')

    # Decoder
    decoder_input = tf.keras.layers.Input(shape=(latent_dim,), name='decoder_input')
    decoded = tf.keras.layers.Dense(256, activation='relu')(decoder_input)
    decoded = tf.keras.layers.Dropout(0.3)(decoded)
    decoded = tf.keras.layers.Dense(512, activation='relu')(decoded)
    reconstruction = tf.keras.layers.Dense(input_dim, activation='sigmoid', name='reconstruction')(decoded)
    
    # We need the standalone decoder model to be returned
    decoder = tf.keras.models.Model(inputs=decoder_input, outputs=reconstruction, name='decoder')

    # Full Autoencoder
    autoencoder_output = decoder(encoder(input_layer))
    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=autoencoder_output, name='autoencoder')
    
    # Return all three models
    return autoencoder, encoder, decoder


def create_classifier(input_dim, num_classes):
    """
    Creates a standard classifier model to be trained on concatenated latent features.
    """
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256, activation='relu')(input_layer)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    classifier = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    return classifier

# Additional Advanced Models

class Sampling(tf.keras.layers.Layer):
    """Custom layer for the reparameterization trick in VAEs."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def create_vae(input_dim, latent_dim=128):
    """Creates a Variational Autoencoder (VAE) model."""
    # Encoder
    encoder_inputs = tf.keras.layers.Input(shape=(input_dim,), name='vae_encoder_input')
    x = tf.keras.layers.Dense(512, activation='relu')(encoder_inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.models.Model(encoder_inputs, [z_mean, z_log_var, z], name='vae_encoder')

    # Decoder
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='vae_decoder_input')
    x = tf.keras.layers.Dense(256, activation='relu')(latent_inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    reconstruction = tf.keras.layers.Dense(input_dim, activation='sigmoid')(x)
    decoder = tf.keras.models.Model(latent_inputs, reconstruction, name='vae_decoder')
    # Full VAE
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = tf.keras.models.Model(encoder_inputs, outputs, name='vae')

    # Add KL divergence loss
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    vae.add_loss(kl_loss)
    
    # Return all three models, consistent with autoencoder
    return vae, encoder, decoder


def create_attention_classifier(num_views, latent_dim_per_view, num_classes):
    """Creates a classifier with an attention mechanism to weigh different omics views."""
    concatenated_dim = num_views * latent_dim_per_view
    input_layer = tf.keras.layers.Input(shape=(concatenated_dim,))
    
    reshaped = tf.keras.layers.Reshape((num_views, latent_dim_per_view))(input_layer)
    attention_output = tf.keras.layers.Attention()([reshaped, reshaped])
    flattened = tf.keras.layers.Flatten()(attention_output)
    
    x = tf.keras.layers.Dense(256, activation='relu')(flattened)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    classifier = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    return classifier