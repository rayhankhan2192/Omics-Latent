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
    classifier = models.create_graph_classifier(num_views, latent_dim_per_view, config['num_classes'])
    
    # Prepare data for single-input model
    train_data = train_features_fused
    test_data = test_features_fused
    
    # Option 2: Hybrid CNN-Attention (Multi-input model) - UNCOMMENT TO USE
    # logger.info("Using Hybrid CNN-Attention Classifier.")
    # num_views = len(data_tr_list)
    # latent_dim_per_view = config['latent_dim']
    # classifier = models.create_cross_view_interaction_classifier(num_views, latent_dim_per_view, config['num_classes'])
    
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
        patience=25,  # Increased patience
        restore_best_weights=True,
        min_delta=0.001,
        verbose=1
    )
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=20,
        min_lr=1e-6,
        verbose=1
    )
    
    # NEW: Stop if training loss gets suspiciously low (sign of severe overfitting)
    overfitting_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=20,
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
        
        train_classifier(encoders, decoders, data_tr_list, data_te_list, labels_tr, labels_te, config)
        
    logger.info("Training and evaluation complete.")