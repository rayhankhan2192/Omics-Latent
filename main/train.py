import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import logging

# Local imports
from main import features
from main import models

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='INFO:%(module)s:%(message)s')


def pretrain_encoders(data_tr_list, config):
    """
    Trains an autoencoder for each omics view in an unsupervised manner.
    """
    encoders = []
    for i, data_tr in enumerate(data_tr_list):
        logging.info(f"... Pre-training autoencoder for view {i+1} ...")
        input_dim = data_tr.shape[1]
        
        # Create the autoencoder and encoder models
        autoencoder, encoder = models.create_autoencoder(input_dim, latent_dim=config['latent_dim'])
        
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate_pretrain']), loss='mse')
        
        # Train the autoencoder
        autoencoder.fit(
            data_tr, data_tr,
            epochs=config['epochs_pretrain'],
            batch_size=config['batch_size'],
            shuffle=True,
            verbose=1
        )
        encoders.append(encoder)
    logging.info("All autoencoders pre-trained successfully.")
    return encoders


def train_classifier(encoders, data_tr_list, data_te_list, labels_tr, labels_te, config):
    """
    Extracts latent features, fuses them, and trains a final classifier.
    """
    # Extract and Fuse Latent Features ---
    logging.info("Extracting and fusing latent features...")
    train_latent_features_list = [encoder.predict(data) for encoder, data in zip(encoders, data_tr_list)]
    test_latent_features_list = [encoder.predict(data) for encoder, data in zip(encoders, data_te_list)]
    
    train_features_fused = np.concatenate(train_latent_features_list, axis=1)
    test_features_fused = np.concatenate(test_latent_features_list, axis=1)
    
    logging.info(f"Shape of fused training features: {train_features_fused.shape}")
    logging.info(f"Shape of fused test features: {test_features_fused.shape}")

    # Train the Classifier ---
    logging.info("Training the final classifier...")
    classifier_input_dim = train_features_fused.shape[1]
    
    # Create and compile the classifier model
    classifier = models.create_classifier(classifier_input_dim, config['num_classes'])
    classifier.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate_classify']),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    #Robust Label Encoding ---
    # Use LabelEncoder to handle any label format and convert them to the required 0-indexed format.
    le = LabelEncoder()
    labels_tr_encoded = le.fit_transform(labels_tr)
    labels_te_encoded = le.transform(labels_te)
    
    classifier.fit(
        train_features_fused,
        labels_tr_encoded,  # Use the new encoded labels
        epochs=config['epochs_classify'],
        batch_size=config['batch_size'],
        shuffle=True,
        validation_data=(test_features_fused, labels_te_encoded), # Use the new encoded labels
        verbose=1
    )

    # Evaluate the Final Model ---
    logging.info("Evaluating the final model on the test set...")
    predictions = np.argmax(classifier.predict(test_features_fused), axis=1)
    
    # Use the encoded labels for calculating metrics
    accuracy = accuracy_score(labels_te_encoded, predictions)
    f1 = f1_score(labels_te_encoded, predictions, average='macro') # 'macro' is good for class imbalance
    
    logging.info("\n--- FINAL RESULTS ---")
    logging.info(f"Accuracy on Test Set: {accuracy:.4f}")
    logging.info(f"Macro F1-Score on Test Set: {f1:.4f}")
    logging.info("---------------------\n")


def run(config):
    """
    Main function to run the complete training and evaluation pipeline.
    """
    logging.info(f"Using configuration for {config['data_path'].split(os.sep)[-1]} dataset.")
    
    # Load and preprocess data
    data_tr_list, data_te_list, labels_tr, labels_te = features.load_and_preprocess_data(config['data_path'], config['view_list'])
    
    if data_tr_list: # Check if data was loaded successfully
        # 1. Pre-train encoders on each view
        encoders = pretrain_encoders(data_tr_list, config)
        
        # 2. Train the final classifier on the fused latent features
        train_classifier(encoders, data_tr_list, data_te_list, labels_tr, labels_te, config)
        
    logging.info("Training and evaluation complete.")