import tensorflow as tf

def create_autoencoder(input_dim, latent_dim=128, sparsity_l1_reg=None):
    """
    Creates a standard Autoencoder model consisting of an Encoder and a Decoder.
    
    [NEW] Can be a Sparse Autoencoder if sparsity_l1_reg is provided.

    Args:
        input_dim (int): The number of input features.
        latent_dim (int): The dimension of the latent (encoded) space.
        sparsity_l1_reg (float, optional): L1 regularization factor for sparsity. Defaults to None.

    Returns:
        tuple: A tuple containing:
            (autoencoder_model, encoder_model, decoder_model)
    """
    
    # Create the regularizer if provided
    l1_regularizer = tf.keras.regularizers.l1(sparsity_l1_reg) if sparsity_l1_reg else None
    
    # Encoder 
    input_layer = tf.keras.layers.Input(shape=(input_dim,), name='encoder_input')
    encoded = tf.keras.layers.Dense(512, activation='relu')(input_layer)
    encoded = tf.keras.layers.Dropout(0.3)(encoded)
    encoded = tf.keras.layers.Dense(256, activation='relu')(encoded)
    
    # Add the activity regularizer to the latent space
    latent_space = tf.keras.layers.Dense(
        latent_dim, 
        activation='relu', 
        name='latent_space',
        activity_regularizer=l1_regularizer  # This makes it a Sparse AE
    )(encoded)
    
    encoder = tf.keras.models.Model(inputs=input_layer, outputs=latent_space, name='encoder')

    # --- Decoder ---
    decoder_input = tf.keras.layers.Input(shape=(latent_dim,), name='decoder_input')
    decoded = tf.keras.layers.Dense(256, activation='relu')(decoder_input)
    decoded = tf.keras.layers.Dropout(0.3)(decoded)
    decoded = tf.keras.layers.Dense(512, activation='relu')(decoded)
    reconstruction = tf.keras.layers.Dense(input_dim, activation=None, name='reconstruction')(decoded)
    
    decoder = tf.keras.models.Model(inputs=decoder_input, outputs=reconstruction, name='decoder')

    # Full Autoencoder ---
    autoencoder_output = decoder(encoder(input_layer))
    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=autoencoder_output, name='autoencoder')
    
    return autoencoder, encoder, decoder



def create_classifier(input_dim, num_classes):

    l2_reg = tf.keras.regularizers.l2(1e-4) # Define a standard L2 penalty

    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    
    # Reduced size and added L2 regularization
    x = tf.keras.layers.Dense(
        128, 
        activation='relu', 
        kernel_regularizer=l2_reg
    )(input_layer)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Reduced size and added L2 regularization
    x = tf.keras.layers.Dense(
        64, 
        activation='relu', 
        kernel_regularizer=l2_reg
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    classifier = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    return classifier


# Additional Advanced Models ---

class Sampling(tf.keras.layers.Layer):
    """Custom layer for the reparameterization trick in VAEs."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        # Raw TensorFlow functions are safe to use inside a Layer's .call() method
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Refactor VAE into a custom Model class for Keras 3 compatibility
class VAE(tf.keras.Model):
    """
    Custom VAE Model class that handles its own KL divergence loss.
    This is the standard pattern for Keras 3.
    """
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        # We need to access these by name to calculate the KL loss
        self.z_mean_layer = self.encoder.get_layer('z_mean')
        self.z_log_var_layer = self.encoder.get_layer('z_log_var')

    def call(self, inputs):
        # Forward pass ---
        # 1. Pass input through the encoder
        z_mean, z_log_var, z = self.encoder(inputs)
        
        # 2. Pass the sampled 'z' through the decoder
        reconstruction = self.decoder(z)
        
        # Add KL Divergence Loss ---
        kl_loss_tensor = -0.5 * (1 + z_log_var - tf.keras.ops.square(z_mean) - tf.keras.ops.exp(z_log_var))
        kl_loss = tf.keras.ops.mean(tf.keras.ops.sum(kl_loss_tensor, axis=1))
        self.add_loss(kl_loss)
        
        return reconstruction


def create_vae(input_dim, latent_dim=128, sparsity_l1_reg=None):
    """
    Creates a Variational Autoencoder (VAE) model.
    
    [NEW] Can be a Sparse VAE if sparsity_l1_reg is provided.
    """
    
    # Create the regularizer if provided
    l1_regularizer = tf.keras.regularizers.l1(sparsity_l1_reg) if sparsity_l1_reg else None

    # --- 1. Build the Encoder ---
    encoder_inputs = tf.keras.layers.Input(shape=(input_dim,), name='vae_encoder_input')
    x = tf.keras.layers.Dense(512, activation='relu')(encoder_inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    
    # [NEW] Add activity regularizer to z_mean (the "sparse" part of the latent space)
    z_mean = tf.keras.layers.Dense(
        latent_dim, 
        name='z_mean',
        activity_regularizer=l1_regularizer # This makes it a Sparse VAE
    )(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    # The encoder model outputs all 3 tensors
    encoder = tf.keras.models.Model(encoder_inputs, [z_mean, z_log_var, z], name='vae_encoder')

    # --- 2. Build the Decoder ---
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='vae_decoder_input')
    x = tf.keras.layers.Dense(256, activation='relu')(latent_inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    reconstruction = tf.keras.layers.Dense(input_dim, activation=None)(x)
    decoder = tf.keras.models.Model(latent_inputs, reconstruction, name='vae_decoder')
    
    # --- 3. Build the Full VAE ---
    vae = VAE(encoder, decoder, name='vae')
    
    # Return all three models
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

