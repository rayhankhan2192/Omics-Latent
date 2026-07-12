import tensorflow as tf
import logging

logger = logging.getLogger("Models Module")
logging.basicConfig(level=logging.INFO, format='INFO:%(module)s:%(message)s')

# 1. PRE-TRAINING MODELS (FOR SCENARIOS 1, 2, 3)
def create_autoencoder(input_dim, latent_dim=128, sparsity_l1_reg=None):
    """
    Creates a standard Autoencoder model (Encoder + Decoder).
    Can be a Sparse AE if sparsity_l1_reg is provided.
    """
    l1_regularizer = tf.keras.regularizers.l1(sparsity_l1_reg) if sparsity_l1_reg else None
    
    # Encoder 
    input_layer = tf.keras.layers.Input(shape=(input_dim,), name='encoder_input')
    encoded = tf.keras.layers.Dense(512, activation='relu')(input_layer)
    encoded = tf.keras.layers.Dropout(0.3)(encoded)
    encoded = tf.keras.layers.Dense(256, activation='relu')(encoded)
    latent_space = tf.keras.layers.Dense(
        latent_dim, 
        activation='relu', 
        name='latent_space',
        activity_regularizer=l1_regularizer
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

# 2. CLASSIFIERS FOR SCENARIO 1 (GENERAL / DIRECT FUSION)

def create_graph_classifier(num_views, latent_dim_per_view, num_classes):
    """
    Graph-based classifier using stacked Multi-Head Attention.
    Takes a single FUSED input.
    """
    input_layer = tf.keras.layers.Input(shape=(num_views * latent_dim_per_view,))
    reshaped = tf.keras.layers.Reshape((num_views, latent_dim_per_view))(input_layer)
    
    # Layer 1
    att1 = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=32, dropout=0.3)(reshaped, reshaped)
    att1 = tf.keras.layers.LayerNormalization()(att1 + reshaped) # Residual connection
    
    # Layer 2
    att2 = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=24, dropout=0.3)(att1, att1)
    att2 = tf.keras.layers.LayerNormalization()(att2 + att1) # Residual connection
    
    pooled = tf.keras.layers.GlobalAveragePooling1D()(att2)
    
    # Classifier head
    x = tf.keras.layers.Dense(128, activation='relu', 
                     kernel_regularizer=tf.keras.regularizers.l2(1e-3))(pooled)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.models.Model(inputs=input_layer, outputs=output, name="graph_classifier")

def create_attention_fusion_classifier(num_views, latent_dim_per_view, num_classes):
    """
    Your good working attention fusion classifier.
    Takes a single FUSED input.
    """
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

def create_simple_mlp_classifier(fused_input_dim, num_classes):
    """
    A simple MLP classifier (baseline).
    Takes a single FUSED input.
    """
    input_layer = tf.keras.layers.Input(shape=(fused_input_dim,), name="mlp_input")
    
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(input_layer)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name="simple_mlp_classifier")
    logger.info("Created Simple MLP Classifier")
    return model

# 3. STACKING MODELS (FOR SCENARIOS 2 & 3)

def create_base_classifier(latent_dim, num_classes):
    """
    Level 1 model for SCENARIO 2 (View-Level Stacking).
    Takes a single-view's latent features.
    """
    input_layer = tf.keras.layers.Input(shape=(latent_dim,))
    
    x = tf.keras.layers.Dense(
        32, 
        activation='relu', 
        kernel_regularizer=tf.keras.regularizers.l2(1e-3)
    )(input_layer)
    x = tf.keras.layers.Dropout(0.5)(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name="base_classifier")
    logger.info("Created Simplified Base Classifier (Level 1)")
    return model

def create_meta_classifier(num_views, num_classes):
    """
    Level 2 model for SCENARIO 2 (View-Level Stacking).
    Takes concatenated predictions from base models.
    """
    input_dim_meta = num_views * num_classes
    input_layer = tf.keras.layers.Input(shape=(input_dim_meta,), name="meta_input")
    
    x = tf.keras.layers.Dense(
        16, 
        activation='relu', 
        kernel_regularizer=tf.keras.regularizers.l2(1e-3)
    )(input_layer)
    x = tf.keras.layers.Dropout(0.5)(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name="meta_output")(x)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name="meta_classifier")
    logger.info("Created Simplified Meta Classifier (Level 2)")
    return model

def create_stacking_meta_model(num_base_models, num_classes):
    """
    Level 2 model for SCENARIO 3 (Multi-Model Stacking).
    Takes concatenated predictions from base models.
    """
    input_dim_meta = num_base_models * num_classes
    input_layer = tf.keras.layers.Input(shape=(input_dim_meta,), name="meta_input")
    
    x = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(input_layer)
    x = tf.keras.layers.Dropout(0.5)(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name="meta_output")(x)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name="stacking_meta_classifier")
    logger.info("Created Stacking Meta Classifier (Level 2)")
    return model

# 4. END-TO-END MODEL (FOR SCENARIO 4)

def create_end_to_end_model(input_dims, latent_dim_per_view, num_classes):
    """
    [FIX V3] Reverting to a more balanced L2 penalty (5e-3) and
    slightly lower dropout to allow the model to learn.
    """
    logger.info("Creating End-to-End Multi-Input Classifier (Balanced Regularization)")
    
    input_layers = []
    processed_towers = []
    
    # BALANCED REGULARIZATION IN TOWERS ---
    l2_reg = tf.keras.regularizers.l2(5e-3) # Was 1e-2, now 5e-3
    
    for i, dim in enumerate(input_dims):
        input_layer = tf.keras.layers.Input(shape=(dim,), name=f"view_{i+1}_input")
        
        x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2_reg)(input_layer)
        x = tf.keras.layers.Dropout(0.6)(x) # Stays 0.6
        x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2_reg)(x)
        x = tf.keras.layers.Dropout(0.6)(x) # Stays 0.6
        tower_output = tf.keras.layers.Dense(latent_dim_per_view, activation='relu')(x)
        
        input_layers.append(input_layer)
        processed_towers.append(tower_output)
        
    fused = tf.keras.layers.Concatenate()(processed_towers)
    
    # BALANCED REGULARIZATION IN CLASSIFIER HEAD ---
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2_reg)(fused)
    x = tf.keras.layers.Dropout(0.6)(x) # Was 0.7
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.Dropout(0.6)(x) # Was 0.7
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name="final_output")(x)
    
    model = tf.keras.models.Model(
        inputs=input_layers, 
        outputs=output_layer, 
        name="end_to_end_multi_input_classifier"
    )
    
    return model


# 5. PURE ORIGINAL BASELINE MODEL (FOR SCENARIO 4)

def create_pure_original_baseline(fused_raw_dim, num_classes):
    input_layer = tf.keras.layers.Input(shape=(fused_raw_dim,), name="raw_input")
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-3))(input_layer)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-3))(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name="raw_output")(x)
    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name="pure_original_baseline")

# ==========================================
# 6. ABLATION STUDY ARCHITECTURES
# ==========================================

def create_ablation_hybrid_model(input_dims, latent_dim, num_classes, drop_regularization=False, drop_towers=False):
    """
    Creates specialized hybrid configurations to isolate individual design components.
    """
    input_layers = []
    processed_outputs = []
    
    # Toggle regularization weight parameters
    l2_penalty = 0.0 if drop_regularization else 5e-3
    dropout_rate = 0.0 if drop_regularization else 0.6
    reg_scaffold = tf.keras.regularizers.l2(l2_penalty) if l2_penalty > 0.0 else None
    
    for i, dim in enumerate(input_dims):
        input_layer = tf.keras.layers.Input(shape=(dim,), name=f"ablation_view_{i+1}_input")
        input_layers.append(input_layer)
        
        if drop_towers:
            # Bypass independent feature towers entirely
            processed_outputs.append(input_layer)
        else:
            # Maintain independent view-specific network towers
            x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=reg_scaffold)(input_layer)
            if dropout_rate > 0.0:
                x = tf.keras.layers.Dropout(dropout_rate)(x)
            x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=reg_scaffold)(x)
            if dropout_rate > 0.0:
                x = tf.keras.layers.Dropout(dropout_rate)(x)
            tower_output = tf.keras.layers.Dense(128, activation='relu')(x)
            processed_outputs.append(tower_output)
            
    # Combine feature representations
    fused = tf.keras.layers.Concatenate()(processed_outputs)
    
    # Downstream classification dense layers
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=reg_scaffold)(fused)
    if dropout_rate > 0.0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=reg_scaffold)(x)
    if dropout_rate > 0.0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name="ablation_output")(x)
    
    model = tf.keras.models.Model(inputs=input_layers, outputs=output_layer, name="ablation_hybrid_model")
    return model