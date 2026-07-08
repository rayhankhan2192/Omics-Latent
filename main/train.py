
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from main import features
from main import models

# --- Setup Logging ---
logger = logging.getLogger("Training Module")
logging.basicConfig(level=logging.INFO, format='INFO:%(module)s:%(message)s')


def pretrain_encoders(data_tr_list, config):
    """
    Trains an unsupervised autoencoder for each omics view.
    """
    logger.info("--- Phase 1: Starting Autoencoder Pre-training ---")
    encoders = []
    decoders = [] 
    
    for i, data_tr in enumerate(data_tr_list):
        view_num = config['view_list'][i]
        input_dim = data_tr.shape[1]
        logger.info(f"Training AE for view {view_num} (Input dim: {input_dim})...")
        
        autoencoder, encoder, decoder = models.create_autoencoder(
            input_dim, 
            latent_dim=config['latent_dim'],
            sparsity_l1_reg=config.get('sparsity_l1_reg')
        )
        
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate_pretrain']), loss='mse')
        
        # --- Prepare Data (add noise if Denoising AE) ---
        train_data_input = data_tr
        if config.get('denoising_noise_factor'):
            logger.info(f"Applying denoising noise (factor: {config['denoising_noise_factor']})")
            noise = np.random.normal(loc=0.0, scale=config['denoising_noise_factor'], size=data_tr.shape)
            train_data_input = data_tr + noise

        autoencoder.fit(
            train_data_input, data_tr, # Train on noisy, target clean
            epochs=config['epochs_pretrain'],
            batch_size=config['batch_size'],
            shuffle=True,
            verbose=1
        )
        encoders.append(encoder)
        decoders.append(decoder) 
        
    logger.info("All autoencoders pre-trained successfully.")
    return encoders, decoders


# SCRIPT CONTROLLER


def run_experiment(config):
    """
    Main function to run the complete training and evaluation pipeline
    based on the specified model_type.
    """
    logger.info(f"Loading data for {config['data_path'].split(os.sep)[-1]}...")
    data_tr_list, data_te_list, labels_tr, labels_te = features.load_and_preprocess_data(
        config['data_path'], config['view_list']
    )
    
    if not data_tr_list:
        logger.error("Data loading failed. Exiting.")
        return

    # Encode labels ONCE
    le = LabelEncoder()
    labels_tr_encoded = le.fit_transform(labels_tr)
    labels_te_encoded = le.transform(labels_te)
    class_names = [str(c) for c in le.classes_]
    config['num_classes'] = len(class_names) # Ensure num_classes is correct

    # --- Scenario Controller ---
    
    if config['model_type'] == 'end_to_end':
        # Scenario 4: No pre-training. Train one model from scratch on original data.
        logger.info("--- Running Scenario 4: End-to-End Classifier ---")
        run_end_to_end_classifier(
            data_tr_list, data_te_list, labels_tr_encoded, labels_te_encoded, class_names, config
        )
        
    else:
        # Scenarios 1, 2, 3: All require pre-trained encoders.
        encoders, decoders = pretrain_encoders(data_tr_list, config)
        
        # Extract latent features ONCE
        logger.info("Extracting latent features from pre-trained encoders...")
        train_latent_list, test_latent_list = extract_latent_features(
            encoders, data_tr_list, data_te_list
        )
        
        # Optionally save features
        if config.get('save_features', False):
            save_features(encoders, decoders, data_tr_list, data_te_list, train_latent_list, test_latent_list, config)

        if config['model_type'] == 'general':
            logger.info("--- Running Scenario 1: General (Direct Fusion) Classifier ---")
            run_general_classifier(
                train_latent_list, test_latent_list, labels_tr_encoded, labels_te_encoded, class_names, config
            )
            
        elif config['model_type'] == 'view_stacking':
            logger.info("--- Running Scenario 2: View-Level Stacking Classifier ---")
            run_view_stacking_classifier(
                train_latent_list, test_latent_list, labels_tr_encoded, labels_te_encoded, class_names, config
            )
            
        elif config['model_type'] == 'model_stacking':
            logger.info("--- Running Scenario 3: Multi-Model (Fused) Stacking Classifier ---")
            run_model_stacking_classifier(
                train_latent_list, test_latent_list, labels_tr_encoded, labels_te_encoded, class_names, config
            )
        elif config['model_type'] == 'hybrid_fusion':
            logger.info("--- Running Scenario 5: Hybrid Fused Latent-Original Classifier ---")
            # We must pass the ORIGINAL data (data_tr_list) AND the LATENT data (train_latent_list)
            run_hybrid_fusion_classifier(
                data_tr_list, data_te_list,
                train_latent_list, test_latent_list,
                labels_tr_encoded, labels_te_encoded, class_names, config
            )

    logger.info("Training and evaluation complete.")


# SCENARIO 1: GENERAL (DIRECT FUSION) CLASSIFIER


def run_general_classifier(train_latent_list, test_latent_list, labels_tr_encoded, labels_te_encoded, class_names, config):
    """
    Trains a single classifier on the fused latent features.
    """
    # Fuse features
    train_features_fused = np.concatenate(train_latent_list, axis=1)
    test_features_fused = np.concatenate(test_latent_list, axis=1)
    
    logger.info(f"Shape of fused training features: {train_features_fused.shape}")
    logger.info(f"Shape of fused test features: {test_features_fused.shape}")

    num_views = len(train_latent_list)
    latent_dim_per_view = train_latent_list[0].shape[1]
    fused_dim = train_features_fused.shape[1]
    
    # --- Model Selection ---
    classifier_name = config['classifier']
    logger.info(f"Initializing model: {classifier_name}")
    
    if classifier_name == 'attention':
        classifier = models.create_attention_fusion_classifier(num_views, latent_dim_per_view, config['num_classes'])
    elif classifier_name == 'graph':
        classifier = models.create_graph_classifier(num_views, latent_dim_per_view, config['num_classes'])
    elif classifier_name == 'mlp':
        classifier = models.create_simple_mlp_classifier(fused_dim, config['num_classes'])
    else:
        logger.warning(f"Classifier '{classifier_name}' not recognized. Defaulting to 'attention'.")
        classifier = models.create_attention_fusion_classifier(num_views, latent_dim_per_view, config['num_classes'])

    # Data for single-input models
    train_data = train_features_fused
    test_data = test_features_fused
    
    # --- Compile ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate_classify'])
    classifier.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # --- Callbacks ---
    callbacks = get_default_callbacks()

    # --- Train ---
    logger.info(f"Starting training for {classifier.name}...")
    history = classifier.fit(
        train_data,
        labels_tr_encoded,
        validation_data=(test_data, labels_te_encoded),
        epochs=config["epochs_classify"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )
    
    # --- Evaluate ---
    logger.info(f"Evaluating {classifier.name}...")
    predictions = np.argmax(classifier.predict(test_data), axis=1)
    # evaluate_and_plot(predictions, labels_te_encoded, history, class_names, "general_")
    evaluate_and_plot(predictions, labels_te_encoded, history, class_names, "general_", classifier, test_data)


# SCENARIO 2: VIEW-LEVEL STACKING CLASSIFIER


def run_view_stacking_classifier(train_latent_list, test_latent_list, labels_tr_encoded, labels_te_encoded, class_names, config):
    """
    Trains a stacking classifier where Level 1 models are trained on individual views.
    This method failed in our experiments.
    """
    num_views = len(train_latent_list)
    latent_dim = train_latent_list[0].shape[1]
    num_classes = config['num_classes']
    n_splits = 5
    
    # --- Callbacks (more sensitive for stacking) ---
    callbacks = get_stacking_callbacks()
    
    meta_features_train = []
    meta_features_test = []

    # --- Level 1 Training ---
    for i in range(num_views):
        view_num = config['view_list'][i]
        logger.info(f"--- Training Level 1 Base Classifier for View {view_num} ---")
        
        X_train_view = train_latent_list[i]
        X_test_view = test_latent_list[i]
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_train_preds = np.zeros((X_train_view.shape[0], num_classes))
        fold_test_preds = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_view, labels_tr_encoded)):
            logger.info(f"  ... Fold {fold+1}/{n_splits} for View {view_num}")
            
            X_fold_train, X_fold_val = X_train_view[train_idx], X_train_view[val_idx]
            y_fold_train, y_fold_val = labels_tr_encoded[train_idx], labels_tr_encoded[val_idx]
            
            base_model = models.create_base_classifier(latent_dim, num_classes)
            base_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate_classify'] * 0.1),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            base_model.fit(
                X_fold_train, y_fold_train,
                validation_data=(X_fold_val, y_fold_val),
                epochs=config["epochs_classify"],
                batch_size=config["batch_size"],
                callbacks=callbacks,
                shuffle=True, verbose=0
            )
            oof_train_preds[val_idx] = base_model.predict(X_fold_val)
            fold_test_preds.append(base_model.predict(X_test_view))

        meta_features_train.append(oof_train_preds)
        meta_features_test.append(np.mean(fold_test_preds, axis=0))

    # --- Level 2 Training ---
    logger.info("--- Training Level 2 Meta-Classifier ---")
    X_train_meta = np.concatenate(meta_features_train, axis=1)
    X_test_meta = np.concatenate(meta_features_test, axis=1)
    
    meta_classifier = models.create_meta_classifier(num_views, num_classes)
    meta_classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate_classify'] * 0.05),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = meta_classifier.fit(
        X_train_meta, labels_tr_encoded,
        validation_data=(X_test_meta, labels_te_encoded),
        epochs=config["epochs_classify"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        shuffle=True, verbose=1
    )
    
    # --- Evaluate ---
    logger.info("Evaluating View-Level Stacking model...")
    predictions = np.argmax(meta_classifier.predict(X_test_meta), axis=1)
    # evaluate_and_plot(predictions, labels_te_encoded, history, class_names, "view_stacking_")
    evaluate_and_plot(predictions, labels_te_encoded, history, class_names, "view_stacking_", classifier, test_data)


# SCENARIO 3: MULTI-MODEL (FUSED) STACKING CLASSIFIER


def run_model_stacking_classifier(train_latent_list, test_latent_list, labels_tr_encoded, labels_te_encoded, class_names, config):
    """
    Trains a stacking classifier where Level 1 models are all trained on the FUSED features.
    This method also failed in our experiments.
    """
    # Fuse features
    train_features_fused = np.concatenate(train_latent_list, axis=1)
    test_features_fused = np.concatenate(test_latent_list, axis=1)
    
    num_views = len(train_latent_list)
    latent_dim_per_view = train_latent_list[0].shape[1]
    fused_dim = train_features_fused.shape[1]
    num_classes = config['num_classes']
    n_splits = 5

    # --- Callbacks (more sensitive for stacking) ---
    callbacks = get_stacking_callbacks()
    
    # Define our list of strong Level 1 models
    base_model_creators = {
        "attention_fusion": lambda: models.create_attention_fusion_classifier(
            num_views, latent_dim_per_view, num_classes
        ),
        "graph_classifier": lambda: models.create_graph_classifier(
            num_views, latent_dim_per_view, num_classes
        ),
        "simple_mlp": lambda: models.create_simple_mlp_classifier(
            fused_dim, num_classes
        ),
    }

    meta_features_train = []
    meta_features_test = []

    # --- Level 1 Training ---
    for model_name, create_model in base_model_creators.items():
        logger.info(f"--- Training Level 1 Base Model: {model_name} ---")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_train_preds = np.zeros((train_features_fused.shape[0], num_classes))
        fold_test_preds = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_features_fused, labels_tr_encoded)):
            logger.info(f"  ... Fold {fold+1}/{n_splits} for {model_name}")
            
            X_fold_train, X_fold_val = train_features_fused[train_idx], train_features_fused[val_idx]
            y_fold_train, y_fold_val = labels_tr_encoded[train_idx], labels_tr_encoded[val_idx]
            
            base_model = create_model()
            base_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate_classify'] * 0.1),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            base_model.fit(
                X_fold_train, y_fold_train,
                validation_data=(X_fold_val, y_fold_val),
                epochs=config["epochs_classify"],
                batch_size=config["batch_size"],
                callbacks=callbacks,
                shuffle=True, verbose=0
            )
            oof_train_preds[val_idx] = base_model.predict(X_fold_val)
            fold_test_preds.append(base_model.predict(test_features_fused))

        meta_features_train.append(oof_train_preds)
        meta_features_test.append(np.mean(fold_test_preds, axis=0))

    # --- Level 2 Training ---
    logger.info("--- Training Level 2 Meta-Classifier ---")
    X_train_meta = np.concatenate(meta_features_train, axis=1)
    X_test_meta = np.concatenate(meta_features_test, axis=1)
    
    num_base_models = len(base_model_creators)
    meta_classifier = models.create_stacking_meta_model(num_base_models, num_classes)
    meta_classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate_classify'] * 0.05),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = meta_classifier.fit(
        X_train_meta, labels_tr_encoded,
        validation_data=(X_test_meta, labels_te_encoded),
        epochs=config["epochs_classify"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        shuffle=True, verbose=1
    )
    
    # --- Evaluate ---
    logger.info("Evaluating Multi-Model Stacking model...")
    predictions = np.argmax(meta_classifier.predict(X_test_meta), axis=1)
    # evaluate_and_plot(predictions, labels_te_encoded, history, class_names, "model_stacking_")
    evaluate_and_plot(predictions, labels_te_encoded, history, class_names, "model_stacking_", meta_classifier, X_test_meta)


# SCENARIO 4: END-TO-END (LATENT + ORIGINAL) CLASSIFIER


def run_end_to_end_classifier(data_tr_list, data_te_list, labels_tr_encoded, labels_te_encoded, class_names, config):
    """
    Trains a single, multi-input model from scratch on the *original* data.
    This model learns its *own* latent features and classifies simultaneously.
    """
    
    # Get the input dimensions for each view from the original data
    input_dims = [data.shape[1] for data in data_tr_list]
    num_views = len(data_tr_list)
    
    logger.info("Initializing End-to-End model...")
    # This model is a "multi-input" model
    classifier = models.create_end_to_end_model(
        input_dims, # List of input dimensions [1000, 1000, 503]
        latent_dim_per_view=config['latent_dim'],
        num_classes=config['num_classes']
    )
    
    classifier.summary() # Print the model structure
    
    # Data for multi-input models is a LIST of arrays
    train_data = data_tr_list
    test_data = data_te_list
    
    # --- Compile ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate_classify'])
    classifier.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # --- Callbacks ---
    callbacks = get_default_callbacks()

    # --- Train ---
    logger.info(f"Starting training for {classifier.name}...")
    history = classifier.fit(
        train_data, # This is a list of [view1_tr, view2_tr, view3_tr]
        labels_tr_encoded,
        validation_data=(test_data, labels_te_encoded), # Also a list
        epochs=config["epochs_classify"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )
    
    # --- Evaluate ---
    logger.info(f"Evaluating {classifier.name}...")
    predictions = np.argmax(classifier.predict(test_data), axis=1)
    # evaluate_and_plot(predictions, labels_te_encoded, history, class_names, "end_to_end_")
    evaluate_and_plot(predictions, labels_te_encoded, history, class_names, "end_to_end_", classifier, test_data)


# SCENARIO 5: HYBRID FUSED LATENT-ORIGINAL CLASSIFIER

from sklearn.utils import class_weight
def run_hybrid_fusion_classifier(data_tr_list, data_te_list, 
                                 train_latent_list, test_latent_list,
                                 labels_tr_encoded, labels_te_encoded, class_names, config):
    """
    Trains a multi-input model on *hybrid* features.
    Each input is the concatenation of [Original_View_Data + Latent_View_Data].
    """
    
    num_views = len(data_tr_list)
    
    # --- 1. Create the Hybrid Feature Lists ---
    logger.info("Creating hybrid (latent + original) feature sets...")
    train_hybrid_data = []
    test_hybrid_data = []
    hybrid_input_dims = []

    for i in range(num_views):
        # Concatenate along the feature axis (axis=1)
        tr_hybrid = np.concatenate([data_tr_list[i], train_latent_list[i]], axis=1)
        te_hybrid = np.concatenate([data_te_list[i], test_latent_list[i]], axis=1)
        
        train_hybrid_data.append(tr_hybrid)
        test_hybrid_data.append(te_hybrid)
        
        # Store the new dimension (e.g., 1000 + 128 = 1128)
        hybrid_input_dims.append(tr_hybrid.shape[1])

    logger.info(f"New hybrid input dimensions: {hybrid_input_dims}")
    
    # --- 2. Create the Model ---
    # We can reuse the same model structure as Scenario 4,
    # as it's already a multi-input model.
    # The "VCDN" you mentioned would be implemented here.
    logger.info("Initializing Hybrid Fusion model...")
    classifier = models.create_end_to_end_model(
        input_dims=hybrid_input_dims, # Pass the new hybrid dims
        latent_dim_per_view=config['latent_dim'],
        num_classes=config['num_classes']
    )
    
    classifier.summary()
    
    # The data is the LIST of hybrid arrays
    train_data = train_hybrid_data
    test_data = test_hybrid_data
    
    # --- 3. Compile (with class weights) ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate_classify'])
    classifier.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Calculating class weights...")
    weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(labels_tr_encoded),
        y=labels_tr_encoded
    )
    class_weights_dict = dict(enumerate(weights))
    
    # Manually boost class 0
    #class_weights_dict[0] = class_weights_dict[0] * 2.0 
    logger.info(f"Using class weights: {class_weights_dict}")

    # --- 4. Train ---
    callbacks = get_default_callbacks()
    logger.info(f"Starting training for {classifier.name}...")
    
    history = classifier.fit(
        train_data, # This is a list [hybrid_v1_tr, hybrid_v2_tr, hybrid_v3_tr]
        labels_tr_encoded,
        validation_data=(test_data, labels_te_encoded),
        epochs=config["epochs_classify"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
        class_weight=class_weights_dict
    )
    
    # --- 5. Evaluate ---
    logger.info(f"Evaluating {classifier.name}...")
    predictions = np.argmax(classifier.predict(test_data), axis=1)
    # evaluate_and_plot(predictions, labels_te_encoded, history, class_names, "hybrid_fusion_")
    evaluate_and_plot(predictions, labels_te_encoded, history, class_names, "hybrid_fusion_", classifier, test_data)


# HELPER FUNCTIONS (UTILITIES)

def extract_latent_features(encoders, data_tr_list, data_te_list):
    """Utility to extract latent features using the trained encoders."""
    train_latent_list = []
    test_latent_list = []
    for i, (encoder, data_tr, data_te) in enumerate(zip(encoders, data_tr_list, data_te_list)):
        if encoder.name == 'vae_encoder':
            logger.info(f"Extracting z_mean features from VAE for view {i+1}")
            train_latent_features = encoder.predict(data_tr)[0] # z_mean is output 0
            test_latent_features = encoder.predict(data_te)[0]
        else:
            logger.info(f"Extracting latent features from AE for view {i+1}")
            train_latent_features = encoder.predict(data_tr)
            test_latent_features = encoder.predict(data_te)
        train_latent_list.append(train_latent_features)
        test_latent_list.append(test_latent_features)
    return train_latent_list, test_latent_list

def save_features(encoders, decoders, data_tr_list, data_te_list, train_latent_list, test_latent_list, config):
    """Utility to save latent and reconstructed features."""
    logger.info("Saving latent and reconstructed features...")
    save_path = os.path.join(config['data_path'], 'generated_features')
    os.makedirs(save_path, exist_ok=True)
    
    for i, (decoder, latent_tr, latent_te) in enumerate(zip(decoders, train_latent_list, test_latent_list)):
        view_num = config['view_list'][i]
        logger.info(f"Saving features for view {view_num}...")
        
        pd.DataFrame(latent_tr).to_csv(os.path.join(save_path, f'{view_num}_latent_features_train.csv'), header=False, index=False)
        pd.DataFrame(latent_te).to_csv(os.path.join(save_path, f'{view_num}_latent_features_test.csv'), header=False, index=False)
        
        reconstructed_tr = decoder.predict(latent_tr)
        reconstructed_te = decoder.predict(latent_te)
        pd.DataFrame(reconstructed_tr).to_csv(os.path.join(save_path, f'{view_num}_reconstructed_features_train.csv'), header=False, index=False)
        pd.DataFrame(reconstructed_te).to_csv(os.path.join(save_path, f'{view_num}_reconstructed_features_test.csv'), header=False, index=False)

    logger.info(f"All features saved to {save_path}")

def get_default_callbacks():
    """Returns the standard callbacks for 'general' and 'end_to_end' models."""
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=20,
        min_lr=1e-6,
        verbose=1
    )
    return [early_stopper, lr_scheduler]

def get_stacking_callbacks():
    """Returns more sensitive callbacks for stacking models that overfit fast."""
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", 
        patience=25, # Shorter patience
        restore_best_weights=True, 
        verbose=1
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.5, 
        patience=10, # More aggressive LR reduction
        min_lr=1e-7,
        verbose=1
    )
    return [early_stopper, lr_scheduler]

# def evaluate_and_plot(predictions, labels_te_encoded, history, class_names, file_prefix):
#     """
#     Calculates metrics, prints reports, and saves plots, prefixed with scenario name.
#     """
#     # --- Calculate Metrics ---
#     accuracy = accuracy_score(labels_te_encoded, predictions)
#     f1 = f1_score(labels_te_encoded, predictions, average="macro")
#     report = classification_report(labels_te_encoded, predictions, target_names=class_names, zero_division=0)
    
#     logger.info("\n--- FINAL RESULTS ---")
#     logger.info(f"Accuracy on Test Set: {accuracy:.4f}")
#     logger.info(f"Macro F1-Score on Test Set: {f1:.4f}")
#     logger.info("---------------------\n")
#     logger.info("\n--- Classification Report ---")
#     logger.info(f"\n{report}")
#     logger.info("-----------------------------\n")

#     # --- Confusion Matrix Plot ---
#     cm_filename = f"{file_prefix}confusion_matrix.png"
#     logger.info(f"Generating confusion matrix plot... ({cm_filename})")
#     cm = confusion_matrix(labels_te_encoded, predictions)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.title(f"{file_prefix} Confusion Matrix")
#     plt.ylabel('Actual Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     plt.savefig(cm_filename, dpi=300)
#     plt.close()

#     # === Plot Training History ===
#     curves_filename = f"{file_prefix}training_curves.png"
#     logger.info(f"Generating training curves plot... ({curves_filename})")
#     plt.figure(figsize=(12, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
#     plt.title('Accuracy Curves')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
    
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
#     plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
#     plt.title('Loss Curves')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
    
#     plt.tight_layout()
#     plt.savefig(curves_filename, dpi=300)
#     plt.close()

def evaluate_and_plot(predictions, labels_te_encoded, history, class_names, file_prefix, classifier, test_data):
    """
    Calculates operational metrics, computes threshold-independent AUC-ROC scores,
    prints classification reports, and saves performance plots.
    """
    # --- Calculate Standard Metrics ---
    accuracy = accuracy_score(labels_te_encoded, predictions)
    f1 = f1_score(labels_te_encoded, predictions, average="macro")
    report = classification_report(labels_te_encoded, predictions, target_names=class_names, zero_division=0)
    
    logger.info("\n--- FINAL RESULTS ---")
    logger.info(f"Accuracy on Test Set: {accuracy:.4f}")
    logger.info(f"Macro F1-Score on Test Set: {f1:.4f}")
    
    # --- Compute Probabilities for AUC-ROC ---
    test_probabilities = classifier.predict(test_data)
    
    if test_probabilities.shape[1] == 2:
        # Binary Classification Path for ROSMAP
        # Extract probabilities for the positive class (column 1)
        binary_auc = roc_auc_score(labels_te_encoded, test_probabilities[:, 1])
        logger.info(f"Binary AUC-ROC on Test Set: {binary_auc:.4f}")
    else:
        # Multi-Class Classification Path for BRCA
        # Compute macro-averaged One-vs-Rest AUC-ROC
        multiclass_auc = roc_auc_score(
            labels_te_encoded, 
            test_probabilities, 
            multi_class="ovr", 
            average="macro"
        )
        logger.info(f"Macro One-vs-Rest AUC-ROC on Test Set: {multiclass_auc:.4f}")
        
    logger.info("---------------------\n")
    logger.info("\n--- Classification Report ---")
    logger.info(f"\n{report}")
    logger.info("-----------------------------\n")

    # --- Confusion Matrix Plot ---
    cm_filename = f"{file_prefix}confusion_matrix.png"
    logger.info(f"Generating confusion matrix plot... ({cm_filename})")
    cm = confusion_matrix(labels_te_encoded, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{file_prefix} Confusion Matrix")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(cm_filename, dpi=300)
    plt.close()

    # === Plot Training History ===
    curves_filename = f"{file_prefix}training_curves.png"
    logger.info(f"Generating training curves plot... ({curves_filename})")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(curves_filename, dpi=300)
    plt.close()