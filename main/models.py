import tensorflow as tf
import logging
logger = logging.getLogger("Training Module")
logging.basicConfig(level=logging.INFO, format='INFO:%(module)s:%(message)s')
# Supervised Autoencoder
def create_supervised_autoencoder(input_dim, latent_dim, num_classes, sparsity_l1_reg=None):
    """
    Creates a SUPERVISED Autoencoder.
    This model has two outputs:
    1. 'decoder_output': The standard reconstruction.
    2. 'classifier_output': A classification head on the latent space.
    """
    
    l1_regularizer = tf.keras.regularizers.l1(sparsity_l1_reg) if sparsity_l1_reg else None

    # Build the Encoder
    encoder_inputs = tf.keras.layers.Input(shape=(input_dim,), name='encoder_input')
    x = tf.keras.layers.Dense(512, activation='relu')(encoder_inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    
    z = tf.keras.layers.Dense(
        latent_dim, 
        name='latent_space',
        activity_regularizer=l1_regularizer
    )(x)
    
    # Stand-alone encoder model
    encoder = tf.keras.models.Model(encoder_inputs, z, name='encoder')

    # Build the Decoder 
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='decoder_input')
    x_dec = tf.keras.layers.Dense(256, activation='relu')(latent_inputs)
    x_dec = tf.keras.layers.Dropout(0.3)(x_dec)
    x_dec = tf.keras.layers.Dense(512, activation='relu')(x_dec)
    reconstruction = tf.keras.layers.Dense(input_dim, activation=None, name='decoder_output')(x_dec)
    
    # Stand-alone decoder model
    decoder = tf.keras.models.Model(latent_inputs, reconstruction, name='decoder')
    
    #  Build the Classifier Head
    classifier_output = tf.keras.layers.Dense(
        num_classes, 
        activation='softmax', 
        name='classifier_output'
    )(z)

    # Build the Full Supervised AE
    
    # Connect the encoder's output 'z' to the 'decoder' model
    # This creates the reconstruction output for *this* model
    reconstruction_for_autoencoder = decoder(z)
    
    # The full model has one input and two outputs,
    # both of which are now connected to the input.
    autoencoder = tf.keras.models.Model(
        inputs=encoder_inputs, 
        outputs=[reconstruction_for_autoencoder, classifier_output], 
        name='supervised_autoencoder'
    )
    
    return autoencoder, encoder, decoder

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

    # Decoder 
    decoder_input = tf.keras.layers.Input(shape=(latent_dim,), name='decoder_input')
    decoded = tf.keras.layers.Dense(256, activation='relu')(decoder_input)
    decoded = tf.keras.layers.Dropout(0.3)(decoded)
    decoded = tf.keras.layers.Dense(512, activation='relu')(decoded)
    reconstruction = tf.keras.layers.Dense(input_dim, activation=None, name='reconstruction')(decoded)
    
    decoder = tf.keras.models.Model(inputs=decoder_input, outputs=reconstruction, name='decoder')

    # Full Autoencoder 
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


# Additional Advanced Models 

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
        self.z_mean_layer = self.encoder.get_layer('z_mean')
        self.z_log_var_layer = self.encoder.get_layer('z_log_var')

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        
        reconstruction = self.decoder(z)
        
        kl_loss_tensor = -0.5 * (1 + z_log_var - tf.keras.ops.square(z_mean) - tf.keras.ops.exp(z_log_var))
        kl_loss = tf.keras.ops.mean(tf.keras.ops.sum(kl_loss_tensor, axis=1))
        BETA = 0.01 
        self.add_loss(BETA * kl_loss)
        
        return reconstruction


def create_vae(input_dim, latent_dim=128, sparsity_l1_reg=None):
    
    # Create the regularizer if provided
    l1_regularizer = tf.keras.regularizers.l1(sparsity_l1_reg) if sparsity_l1_reg else None

    # Build the Encoder
    encoder_inputs = tf.keras.layers.Input(shape=(input_dim,), name='vae_encoder_input')
    x = tf.keras.layers.Dense(512, activation='relu')(encoder_inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)

    z_mean = tf.keras.layers.Dense(
        latent_dim, 
        name='z_mean',
        activity_regularizer=l1_regularizer
    )(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    # The encoder model outputs all 3 tensors
    encoder = tf.keras.models.Model(encoder_inputs, [z_mean, z_log_var, z], name='vae_encoder')

    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='vae_decoder_input')
    x = tf.keras.layers.Dense(256, activation='relu')(latent_inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    # Corrected activation: None, to match the StandardScaled data
    reconstruction = tf.keras.layers.Dense(input_dim, activation=None)(x)
    decoder = tf.keras.models.Model(latent_inputs, reconstruction, name='vae_decoder')
    
    vae = VAE(encoder, decoder, name='vae')
    
    # Return all three models
    return vae, encoder, decoder

# classifier models for fused latent features

# Best Performing Advanced Classifier Models
def create_graph_classifier(num_views, latent_dim_per_view, num_classes, num_neighbors=10):
    """
    Graph-based classifier: treats samples as nodes in a similarity graph.
    Requires computing k-nearest neighbors during training.
    """
    # This is more complex and requires additional graph construction
    # Here's a simplified version using attention as proxy for graph edges

    
    input_layer = tf.keras.layers.Input(shape=(num_views * latent_dim_per_view,))
    
    # Reshape to views
    reshaped = tf.keras.layers.Reshape((num_views, latent_dim_per_view))(input_layer)
    
    # Multi-head attention acts like graph message passing
    # Each head learns different relationships
    att1 = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=32, dropout=0.3)(reshaped, reshaped)
    att1 = tf.keras.layers.LayerNormalization()(att1)
    
    # Second layer of "message passing"
    att2 = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=24, dropout=0.3)(att1, att1)
    att2 = tf.keras.layers.LayerNormalization()(att2)
    
    # Aggregate
    pooled = tf.keras.layers.GlobalAveragePooling1D()(att2)
    
    # Classifier
    x = tf.keras.layers.Dense(128, activation='relu', 
                     kernel_regularizer=tf.keras.regularizers.l2(1e-3))(pooled)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.models.Model(inputs=input_layer, outputs=output, name="graph_classifier")


# Old one good working attention fusion classifier
def create_attention_fusion_classifier(num_views, latent_dim_per_view, num_classes):

    input_layer = tf.keras.layers.Input(shape=(num_views * latent_dim_per_view,))
    reshaped = tf.keras.layers.Reshape((num_views, latent_dim_per_view))(input_layer)

    att = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=latent_dim_per_view)(reshaped, reshaped)
    pooled = tf.keras.layers.GlobalAveragePooling1D()(att)

    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(pooled)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=out, name="attention_fusion_classifier")




# Tested simpler attention classifier

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

# new optimized version

def create_attention_fusion_classifier2(num_views, latent_dim_per_view, num_classes):
    """
    Optimized attention fusion classifier with view-specific processing.
    Reduces overfitting through architectural improvements.
    """
    input_layer = tf.keras.layers.Input(shape=(num_views * latent_dim_per_view,))
    reshaped = tf.keras.layers.Reshape((num_views, latent_dim_per_view))(input_layer)
    
    # Project each view to a common lower-dimensional space first
    view_projection = tf.keras.layers.Dense(
        128, 
        activation='relu',
        #kernel_regularizer=tf.keras.regularizers.l2(1e-3)
    )
    projected = tf.keras.layers.TimeDistributed(view_projection)(reshaped)
    
    # Single-head attention with reduced key_dim
    att = tf.keras.layers.MultiHeadAttention(
        num_heads=1, 
        key_dim=32,
        dropout=0.5
    )(projected, projected)
    pooled = tf.keras.layers.GlobalAveragePooling1D()(att)
    
    # Much simpler classifier head
    x = tf.keras.layers.Dropout(0.5)(pooled)
    x = tf.keras.layers.Dense(
        32, 
        activation='relu', 
        kernel_regularizer=tf.keras.regularizers.l2(1e-3)
    )(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=out, name="attention_fusion_classifier")

# Multi-Branch 1D-CNN Classifier
def create_multi_branch_cnn_classifier(num_views, latent_dim_per_view, num_classes):

    def create_cnn_branch(input_shape, name):
        """Helper function to create a single, simple 1D-CNN branch."""
        inp = tf.keras.layers.Input(shape=input_shape, name=name)
        x = tf.keras.layers.Reshape((input_shape[0], 1))(inp)
        x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        return tf.keras.models.Model(inp, x, name=f"{name}_branch")

    inputs = [tf.keras.layers.Input(shape=(latent_dim_per_view,), name=f"view_{i+1}") for i in range(num_views)]
    
    branches = [create_cnn_branch((latent_dim_per_view,), f"cnn_view_{i+1}") for i in range(num_views)]
    
    # Apply each input to its corresponding independent branch
    processed = [branches[i](inputs[i]) for i in range(num_views)]
    
    # Concatenate the outputs from all branches
    fused = tf.keras.layers.Concatenate()(processed)
    
    # Simple MLP head for final classification
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fused)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Return the complete multi-input model
    return tf.keras.models.Model(inputs=inputs, outputs=out, name="light_cnn_classifier")


def create_hybrid_cnn_attention_classifier(num_views, latent_dim_per_view, num_classes):
    """
    Combines the proven Multi-Branch CNN architecture with attention fusion.
    Best of both worlds: CNN extracts view-specific patterns, attention weighs importance.
    """
    inputs = []
    branch_outputs = []
    
    # Process each omics view with its own lightweight CNN branch
    for i in range(num_views):
        inp = tf.keras.layers.Input(shape=(latent_dim_per_view,), name=f'view_{i+1}')
        inputs.append(inp)
        
        # Reshape for 1D-CNN: (batch, features, channels)
        x = tf.keras.layers.Reshape((latent_dim_per_view, 1))(inp)
        
        # Lightweight 1D-CNN to find local patterns within this omics view
        x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)  # (batch, 16)
        
        branch_outputs.append(x)
    
    # Stack the branch outputs: (batch, num_views, 16)
    stacked = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(branch_outputs)
    
    # Apply attention to learn view importance
    att = tf.keras.layers.MultiHeadAttention(
        num_heads=1, 
        key_dim=8,
        dropout=0.2
    )(stacked, stacked)
    pooled = tf.keras.layers.GlobalAveragePooling1D()(att)
    
    # Simple classification head
    x = tf.keras.layers.Dropout(0.5)(pooled)
    x = tf.keras.layers.Dense(
        24, 
        activation='relu', 
        kernel_regularizer=tf.keras.regularizers.l2(1e-3)
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.models.Model(inputs=inputs, outputs=out, name="hybrid_cnn_attention")


# Mixture of Experts (MoE) 

def create_optimized_moe_classifier(num_views, latent_dim_per_view, num_classes):
    
    inputs = []
    expert_outputs = []
    
    # Create deeper, more capable experts for each view
    for i in range(num_views):
        inp = tf.keras.layersers.Input(shape=(latent_dim_per_view,), name=f'view_{i+1}')
        inputs.append(inp)
        
        # Expert network with better capacity
        expert = tf.keras.layers.Dense(64, activation='relu', 
                             kernel_regularizer=tf.keras.regularizers.l2(5e-4))(inp)
        expert = tf.keras.layers.BatchNormalization()(expert)
        expert = tf.keras.layers.Dropout(0.3)(expert)
        
        expert = tf.keras.layers.Dense(32, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(5e-4))(expert)
        expert = tf.keras.layers.BatchNormalization()(expert)
        expert = tf.keras.layers.Dropout(0.2)(expert)
        
        expert_outputs.append(expert)
    
    # Stack expert outputs: (batch, num_views, 32)
    stacked_experts = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(expert_outputs)
    
    # Enhanced gating network with attention
    concatenated = tf.keras.layers.Concatenate()(inputs)
    
    # First compute importance scores
    gate = tf.keras.layers.Dense(num_views * 8, activation='relu',
                       kernel_regularizer=tf.keras.regularizers.l2(5e-4))(concatenated)
    gate = tf.keras.layers.Dropout(0.2)(gate)
    gate = tf.keras.layers.Dense(num_views, activation='softmax', name='gating_weights')(gate)
    gate = tf.keras.layers.Reshape((num_views, 1))(gate)
    
    # Weighted combination of experts
    weighted_experts = tf.keras.layers.Multiply()([stacked_experts, gate])
    gated_output = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted_experts)
    
    # Also use direct concatenation (ensemble approach)
    concat_experts = tf.keras.layers.Concatenate()(expert_outputs)
    
    # Combine gated and concatenated paths
    combined = tf.keras.layers.Concatenate()([gated_output, concat_experts])
    
    # Final classifier with good capacity
    x = tf.keras.layers.Dense(64, activation='relu', 
                     kernel_regularizer=tf.keras.regularizers.l2(5e-4))(combined)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(32, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.models.Model(inputs=inputs, outputs=output, name="optimized_moe")


# Ensemble of Shallow Models 
def create_ensemble_classifier(num_views, latent_dim_per_view, num_classes):
    """
    Ensemble of multiple shallow architectures.
    Works well on small datasets by reducing variance.
    """
    
    input_layer = tf.keras.layers.Input(shape=(num_views * latent_dim_per_view,))
    
    # Simple MLP
    mlp = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(input_layer)
    mlp = tf.keras.layers.Dropout(0.4)(mlp)
    mlp = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(mlp)
    
    # Feature attention
    reshaped = tf.keras.layers.Reshape((num_views, latent_dim_per_view))(input_layer)
    att = tf.keras.layers.MultiHeadAttention(num_heads=1, key_dim=32)(reshaped, reshaped)
    att = tf.keras.layers.GlobalAveragePooling1D()(att)
    att = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(att)
    
    # Gated fusion
    gate = tf.keras.layers.Dense(num_views, activation='softmax')(tf.keras.layers.Flatten()(reshaped))
    gate = tf.keras.layers.RepeatVector(latent_dim_per_view)(gate)
    gate = tf.keras.layers.Permute((2, 1))(gate)
    gated = tf.keras.layers.Multiply()([reshaped, gate])
    gated = tf.keras.layers.Flatten()(gated)
    gated = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(gated)
    
    # Ensemble fusion
    ensemble = tf.keras.layers.Concatenate()([mlp, att, gated])
    ensemble = tf.keras.layers.Dropout(0.5)(ensemble)
    ensemble = tf.keras.layers.Dense(48, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(ensemble)
    ensemble = tf.keras.layers.Dropout(0.3)(ensemble)
    
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(ensemble)
    
    return tf.keras.models.Model(inputs=input_layer, outputs=output, name="ensemble_classifier")

# Cross-View Feature Interaction Network
def create_cross_view_interaction_classifier(num_views, latent_dim_per_view, num_classes):
    """
    Models pairwise interactions between omics views.
    Useful when different omics have synergistic effects.
    """
    from tensorflow.keras import layers, models
    
    inputs = []
    view_embeddings = []
    
    # Embed each view to a common space
    for i in range(num_views):
        inp = layers.Input(shape=(latent_dim_per_view,), name=f'view_{i+1}')
        inputs.append(inp)
        
        # Project to common embedding space
        emb = layers.Dense(32, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-3))(inp)
        emb = layers.BatchNormalization()(emb)
        view_embeddings.append(emb)
    
    # Compute pairwise interactions
    interactions = []
    for i in range(num_views):
        for j in range(i + 1, num_views):
            # Element-wise product captures feature interactions
            interaction = layers.Multiply()([view_embeddings[i], view_embeddings[j]])
            interactions.append(interaction)
    
    # Combine individual views and interactions
    all_features = view_embeddings + interactions
    combined = layers.Concatenate()(all_features)
    
    # Classification head
    x = layers.Dropout(0.5)(combined)
    x = layers.Dense(48, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(24, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=output, name="cross_view_interaction")


# ABayesian Neural Network (Uncertainty-aware)
import tensorflow_probability as tfp

def create_bayesian_classifier(num_views, latent_dim_per_view, num_classes):
    """
    Bayesian Neural Network with uncertainty quantification.
    Each prediction comes with a confidence measure.
    """
    from tensorflow.keras import layers, models
    
    input_layer = layers.Input(shape=(num_views * latent_dim_per_view,))
    
    # Reshape for view-wise processing
    reshaped = layers.Reshape((num_views, latent_dim_per_view))(input_layer)
    
    # Simple view processing
    view_proj = layers.Dense(32, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-3))
    projected = layers.TimeDistributed(view_proj)(reshaped)
    pooled = layers.GlobalAveragePooling1D()(projected)
    
    # Bayesian layers (use variational inference)
    x = tfp.layers.DenseVariational(
        48,
        make_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
        make_prior_fn=tfp.layers.default_multivariate_normal_fn,
        kl_weight=1/600,  # 1/num_train_samples
        activation='relu'
    )(pooled)
    x = layers.Dropout(0.4)(x)
    
    x = tfp.layers.DenseVariational(
        24,
        make_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
        make_prior_fn=tfp.layers.default_multivariate_normal_fn,
        kl_weight=1/600,
        activation='relu'
    )(x)
    x = layers.Dropout(0.3)(x)
    
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=input_layer, outputs=output, name="bayesian_classifier")

from sklearn.metrics import accuracy_score, f1_score
logger = logging.getLogger(__name__)
def train_ml_ensemble_classifier(train_features, test_features, labels_tr_encoded, labels_te_encoded):
    """
    Ensemble of traditional ML models.
    Often outperforms DL on small datasets.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import VotingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    
    # Define base models
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, 
                                min_samples_split=10, random_state=42)
    xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=42)
    lgbm = LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                         random_state=42, verbose=-1)
    lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    
    # Voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb),
            ('lgbm', lgbm),
            ('lr', lr)
        ],
        voting='soft'  # Use probability averaging
    )
    
    logger.info("Training ML Ensemble...")
    ensemble.fit(train_features, labels_tr_encoded)
    
    # Predictions
    predictions = ensemble.predict(test_features)
    accuracy = accuracy_score(labels_te_encoded, predictions)
    f1 = f1_score(labels_te_encoded, predictions, average='macro')
    
    logger.info(f"\n--- ML ENSEMBLE RESULTS ---")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    
    return ensemble, accuracy, f1

import tensorflow as tf

def create_regularized_mlp_classifier(input_dim, num_classes, l2=1e-4, dropout=0.5):
    """
    Simple, robust classifier:
      - BatchNorm → stabilizes features
      - Two small Dense layers with L2
      - High dropout to curb overfitting
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name="regularized_mlp_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"),
                 tf.keras.metrics.AUC(name="auc", multi_label=True)]
    )
    return model
