import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import logging

from main import features
from main import models
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger("Training Module")
# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='INFO:%(module)s:%(message)s')

# def pretrain_encoders(data_tr_list, labels_tr, config):
#     """
#     Trains a SUPERVISED autoencoder for each view in the dataset.
#     This forces the latent space to be good for classification.
#     """
#     logger.info("--- Phase 1: Starting SUPERVISED Autoencoder Pre-training ---")
#     encoders = []
#     decoders = []
    
#     # Get pre-training specific hyperparameters
#     lr = config['learning_rate_pretrain']
#     epochs = config['epochs_pretrain']
#     batch_size = config['batch_size']
#     l1_reg = config.get('sparsity_l1', None)
#     noise_factor = config.get('denoising_noise', 0.0)
    
#     # [NEW] Set a weight for the classification loss.
#     # This is a key hyperparameter to tune.
#     # We care more about reconstruction (1.0) than classification (e.g., 0.5)
#     classification_weight = 0.5 

#     for i, data_tr in enumerate(data_tr_list):
#         view_num = config['view_list'][i]
#         input_dim = data_tr.shape[1]
#         logger.info(f"Training Supervised AE for view {view_num} (Input dim: {input_dim})...")
        
#         # --- Model Creation ---
#         # [FIX] Call the new supervised model
#         autoencoder, encoder, decoder = models.create_supervised_autoencoder(
#             input_dim, 
#             latent_dim=config['latent_dim'],
#             num_classes=config['num_classes'], # Pass num_classes
#             sparsity_l1_reg=l1_reg
#         )

#         # --- Compile ---
#         # [FIX] Compile with two losses and weights
#         autoencoder.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
#             loss={
#                 'decoder': 'mse', # <-- CORRECTED
#                 'classifier_output': 'sparse_categorical_crossentropy'
#             },
#             loss_weights={
#                 'decoder': 1.0, # <-- CORRECTED
#                 'classifier_output': classification_weight
#             },
#             metrics={'classifier_output': 'accuracy'} # Optional: monitor accuracy
#         )

#         # --- Prepare Data (add noise if Denoising AE) ---
#         train_data_input = data_tr
#         if noise_factor > 0:
#             logger.info(f"Applying denoising noise (factor: {noise_factor})")
#             noise = np.random.normal(loc=0.0, scale=noise_factor, size=data_tr.shape)
#             train_data_input = data_tr + noise

#         # --- Train ---
#         # [FIX] Fit the model with TWO targets
#         autoencoder.fit(
#             train_data_input,  # Input (potentially noisy)
#             {
#                 'decoder': data_tr,       # Target 1: clean data (CORRECTED)
#                 'classifier_output': labels_tr   # Target 2: the labels
#             },
#             epochs=epochs,
#             batch_size=batch_size,
#             shuffle=True,
#             verbose=2
#         )
        
#         encoders.append(encoder)
#         decoders.append(decoder)
#         logger.info(f"View {view_num} pre-training complete.")

#     logger.info("--- All Supervised Autoencoders Pre-trained Successfully ---")
#     return encoders, decoders


def pretrain_encoders(data_tr_list, config):
    """
    Trains an autoencoder for each omics view in an unsupervised manner.
    """
    encoders = []
    decoders = [] 
    
    for i, data_tr in enumerate(data_tr_list):
        logging.info(f"... Pre-training autoencoder for view {i+1} ...")
        input_dim = data_tr.shape[1]
        
        autoencoder, encoder, decoder = models.create_autoencoder(input_dim, latent_dim=config['latent_dim'])
        
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate_pretrain']), loss='mse')
        
        autoencoder.fit(
            data_tr, data_tr,
            epochs=config['epochs_pretrain'],
            batch_size=config['batch_size'],
            shuffle=True,
            verbose=1
        )
        encoders.append(encoder)
        decoders.append(decoder) 
        
    logging.info("All autoencoders pre-trained successfully.")
    
    return encoders, decoders


def train_classifier(encoders, decoders, data_tr_list, data_te_list, labels_tr, labels_te, config):
    """
    Extracts latent features, fuses them, and trains a final classifier.
    """
    logger.info("Extracting and fusing latent features...")
    
    train_latent_features_list = []
    test_latent_features_list = []

    for i, (encoder, data_tr, data_te) in enumerate(zip(encoders, data_tr_list, data_te_list)):
        
        if encoder.name == 'vae_encoder':
            logger.info(f"Extracting z_mean features from VAE for view {i+1}")
            train_latent_features = encoder.predict(data_tr)[0]
            test_latent_features = encoder.predict(data_te)[0]
        else:
            logger.info(f"Extracting latent features from AE for view {i+1}")
            train_latent_features = encoder.predict(data_tr)
            test_latent_features = encoder.predict(data_te)
            
        train_latent_features_list.append(train_latent_features)
        test_latent_features_list.append(test_latent_features)
        
    # Save features if requested 
    if config.get('save_features', False):
        logger.info("Saving latent and reconstructed features...")
        save_path = os.path.join(config['data_path'], 'generated_features')
        os.makedirs(save_path, exist_ok=True)
        
        for i, (decoder, latent_tr, latent_te) in enumerate(zip(decoders, train_latent_features_list, test_latent_features_list)):
            view_num = config['view_list'][i]
            logger.info(f"Saving features for view {view_num}...")
            
            pd.DataFrame(latent_tr).to_csv(os.path.join(save_path, f'{view_num}_latent_features_train.csv'), header=False, index=False)
            pd.DataFrame(latent_te).to_csv(os.path.join(save_path, f'{view_num}_latent_features_test.csv'), header=False, index=False)
            
            reconstructed_tr = decoder.predict(latent_tr)
            reconstructed_te = decoder.predict(latent_te)
            pd.DataFrame(reconstructed_tr).to_csv(os.path.join(save_path, f'{view_num}_reconstructed_features_train.csv'), header=False, index=False)
            pd.DataFrame(reconstructed_te).to_csv(os.path.join(save_path, f'{view_num}_reconstructed_features_test.csv'), header=False, index=False)

        logger.info(f"All features saved to {save_path}")

    train_features_fused = np.concatenate(train_latent_features_list, axis=1)
    test_features_fused = np.concatenate(test_latent_features_list, axis=1)
    
    logger.info(f"Shape of fused training features: {train_features_fused.shape}")
    logger.info(f"Shape of fused test features: {test_features_fused.shape}")

    # --- Train the Classifier ---
    logger.info("Training the final classifier...")
    
    
    # Option 1: Enhanced Attention Fusion (Single-input model)
    logger.info("Using Enhanced Attention Fusion Classifier.")
    num_views = len(data_tr_list)
    latent_dim_per_view = config['latent_dim']
    classifier = models.create_attention_fusion_classifier(num_views, latent_dim_per_view, config['num_classes'])
    
    # Prepare data for single-input model
    train_data = train_features_fused
    test_data = test_features_fused
    
    # Option 2: Hybrid CNN-Attention (Multi-input model) - UNCOMMENT TO USE
    # logger.info("Using Hybrid CNN-Attention Classifier.")
    # num_views = len(data_tr_list)
    # latent_dim_per_view = config['latent_dim']
    # classifier = models.create_multi_branch_cnn_classifier(num_views, latent_dim_per_view, config['num_classes'])
    
    # # Prepare data for multi-input model
    # train_data = train_latent_features_list
    # test_data = test_latent_features_list

    # === Encode Labels ===
    le = LabelEncoder()
    labels_tr_encoded = le.fit_transform(labels_tr)
    labels_te_encoded = le.transform(labels_te)

    # === Enhanced Callbacks ===
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=40,  # Increased patience
        restore_best_weights=True,
        min_delta=0.0001,
        verbose=1
    )
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=30,
        min_lr=1e-6,
        verbose=1
    )
    
    # NEW: Stop if training loss gets suspiciously low (sign of severe overfitting)
    overfitting_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=30,
        mode='min',
        baseline=0.05,  # If training loss < 0.05, likely overfitting
        restore_best_weights=False,
        verbose=1
    )

    # === Compile with Lower Learning Rate ===
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate_classify'] * 0.5)  # Half the LR
    classifier.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # === Train WITHOUT adding extra noise (the model already has enough regularization) ===
    logger.info("Starting classifier training...")
    history = classifier.fit(
        train_data,
        labels_tr_encoded,
        validation_data=(test_data, labels_te_encoded),
        epochs=config["epochs_classify"],
        batch_size=config["batch_size"],
        callbacks=[early_stopper, lr_scheduler, overfitting_stopper],
        shuffle=True,
        verbose=1
    )

    # === Evaluate ===
    logger.info("Evaluating the final model (from best epoch) on the test set...")
    predictions = np.argmax(classifier.predict(test_data), axis=1)
    accuracy = accuracy_score(labels_te_encoded, predictions)
    f1 = f1_score(labels_te_encoded, predictions, average="macro")

    logger.info("\n--- FINAL RESULTS ---")
    logger.info(f"Accuracy on Test Set: {accuracy:.4f}")
    logger.info(f"Macro F1-Score on Test Set: {f1:.4f}")
    logger.info("---------------------\n")

    # === Plot Training History ===
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300)
    plt.close()

    classifier.save("final_attention_fusion_classifier.keras")
    logger.info("Model saved successfully.")


def run(config):
    """
    Main function to run the complete training and evaluation pipeline.
    """
    logger.info(f"Using configuration for {config['data_path'].split(os.sep)[-1]} dataset.")
    
    data_tr_list, data_te_list, labels_tr, labels_te = features.load_and_preprocess_data(config['data_path'], config['view_list'])
    
    if data_tr_list:
        encoders, decoders = pretrain_encoders(data_tr_list, config)
        #encoders, decoders = pretrain_encoders(data_tr_list, labels_tr, config)
        
        train_classifier(encoders, decoders, data_tr_list, data_te_list, labels_tr, labels_te, config)
        
    logger.info("Training and evaluation complete.")