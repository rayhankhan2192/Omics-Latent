import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import logging

# Local imports
from main import features
from main import models

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='INFO:%(name)s:%(message)s')


def pretrain_encoders(data_tr_list, config):
    """
    Trains an autoencoder for each omics view in an unsupervised manner.
    """
    encoders = []
    decoders = []
    
    noise_factor = config.get('denoising_noise_factor', None)
    l1_reg = config.get('sparsity_l1_reg', None)
    
    if noise_factor:
        logger.info(f"--- Denoising Autoencoder enabled (Noise Factor: {noise_factor}) ---")
    if l1_reg:
        logger.info(f"--- Sparse Autoencoder enabled (L1 Penalty: {l1_reg}) ---")

    for i, data_tr in enumerate(data_tr_list):
        logger.info(f"... Pre-training autoencoder for view {i+1} ...")
        input_dim = data_tr.shape[1]
        
        # Option 1: Standard (or Sparse/Denoising) Autoencoder 
        autoencoder, encoder, decoder = models.create_autoencoder(
            input_dim, 
            latent_dim=config['latent_dim'],
            sparsity_l1_reg=l1_reg
        )
        
        # Option 2: Variational (or Sparse/Denoising) Autoencoder 
        # autoencoder, encoder, decoder = models.create_vae(
        #     input_dim, 
        #     latent_dim=config['latent_dim'],
        #     sparsity_l1_reg=l1_reg
        # )
        
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate_pretrain']), loss='mse')
        
        if noise_factor and noise_factor > 0:
            # Create a "noisy" version of the input data
            data_tr_noisy = tf.keras.layers.Dropout(noise_factor)(data_tr, training=True)
            
            logger.info("Training Denoising AE: Input=Noisy Data, Target=Clean Data")
            autoencoder.fit(
                data_tr_noisy,
                data_tr,
                epochs=config['epochs_pretrain'],
                batch_size=config['batch_size'],
                shuffle=True,
                verbose=1
            )
        else:
            logger.info("Training Standard/Sparse AE: Input=Clean Data, Target=Clean Data")
            autoencoder.fit(
                data_tr,
                data_tr,
                epochs=config['epochs_pretrain'],
                batch_size=config['batch_size'],
                shuffle=True,
                verbose=1
            )
        
        encoders.append(encoder)
        decoders.append(decoder)
        
    logger.info("All autoencoders pre-trained successfully.")
    return encoders, decoders


def train_classifier(encoders, decoders, data_tr_list, data_te_list, labels_tr, labels_te, config):
    """
    Extracts latent features, fuses them, and trains a final classifier.
    """
    # --- Extract and Fuse Latent Features ---
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
    classifier_input_dim = train_features_fused.shape[1]
    
    # Option 1: Standard Classifier (Now with L2 reg)
    # logger.info("Using standard MLP classifier with L2 regularization.")
    # classifier = models.create_classifier(classifier_input_dim, config['num_classes'])
    
    # --- Option 2: Attention Classifier ---
    # [NEW] Switched to Attention Classifier as the default
    logger.info("Using Attention classifier.")
    num_views = len(data_tr_list)
    latent_dim_per_view = config['latent_dim']
    classifier = models.create_attention_classifier(num_views, latent_dim_per_view, config['num_classes'])

    classifier.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate_classify']),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    # Robust Label Encoding 
    le = LabelEncoder()
    labels_tr_encoded = le.fit_transform(labels_tr)
    labels_te_encoded = le.transform(labels_te)

    # Increased patience from 15 to 25
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        verbose=1,
        restore_best_weights=True
    )
    # This layer is only "active" during training
    noise_layer = tf.keras.layers.GaussianNoise(0.1)
    train_features_fused_noisy = noise_layer(train_features_fused, training=True)
    
    classifier.fit(
        train_features_fused_noisy, 
        labels_tr_encoded,
        epochs=config['epochs_classify'],
        batch_size=config['batch_size'],
        shuffle=True,
        validation_data=(test_features_fused, labels_te_encoded), 
        verbose=1,
        callbacks=[early_stopper] 
    )

    logger.info("Evaluating the final model (from best epoch) on the test set...")
    predictions = np.argmax(classifier.predict(test_features_fused), axis=1)
    
    accuracy = accuracy_score(labels_te_encoded, predictions)
    f1 = f1_score(labels_te_encoded, predictions, average='macro')
    
    logger.info("\n--- FINAL RESULTS ---")
    logger.info(f"Accuracy on Test Set: {accuracy:.4f}")
    logger.info(f"Macro F1-Score on Test Set: {f1:.4f}")
    logger.info("---------------------\n")


def run(config):
    """
    Main function to run the complete training and evaluation pipeline.
    """
    logger.info(f"Using configuration for {config['data_path'].split(os.sep)[-1]} dataset.")
    
    data_tr_list, data_te_list, labels_tr, labels_te = features.load_and_preprocess_data(config['data_path'], config['view_list'])
    
    if data_tr_list:
        encoders, decoders = pretrain_encoders(data_tr_list, config)
        
        train_classifier(encoders, decoders, data_tr_list, data_te_list, labels_tr, labels_te, config)
        
    logger.info("Training and evaluation complete.")

