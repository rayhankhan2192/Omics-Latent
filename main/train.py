import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Training Module")
from . import features
from . import models

def pretrain_encoders(data_tr_list, config):
    """
    Pre-trains an autoencoder for each omics view to learn feature representations.

    Args:
        data_tr_list (list): List of training data arrays for each view.
        config (dict): A dictionary containing training configuration parameters.

    Returns:
        list: A list of trained standalone encoder models.
    """
    encoders = []
    for i, data_tr in enumerate(data_tr_list):
        input_dim = data_tr.shape[1]
        autoencoder, encoder = models.create_autoencoder(input_dim, latent_dim=config['latent_dim'])
        autoencoder.compile(optimizer='adam', loss='mse')
        
        autoencoder.fit(data_tr, data_tr,
                        epochs=config['pretrain_epochs'],
                        batch_size=config['batch_size'],
                        shuffle=True,
                        validation_split=0.1,
                        verbose=2)
        
        encoders.append(encoder)
        logger.info(f"Autoencoder for view {i+1} pre-trained successfully.")
    return encoders

def train_classifier(encoders, data_tr_list, data_te_list, labels_tr, labels_te, config):
    """
    Trains and evaluates a classifier on the latent features extracted by the encoders.

    Args:
        encoders (list): List of pre-trained encoder models.
        data_tr_list (list): List of training data arrays for each view.
        data_te_list (list): List of testing data arrays for each view.
        labels_tr (np.array): Training labels.
        labels_te (np.array): Testing labels.
        config (dict): A dictionary containing training configuration parameters.
    """
    # Extract latent features by transforming data with the pre-trained encoders
    train_latent_list = [encoder.predict(data_tr) for encoder, data_tr in zip(encoders, data_tr_list)]
    test_latent_list = [encoder.predict(data_te) for encoder, data_te in zip(encoders, data_te_list)]

    # Concatenate latent features from all views
    train_latent_features = np.concatenate(train_latent_list, axis=1)
    test_latent_features = np.concatenate(test_latent_list, axis=1)

    logger.info(f"Concatenated training latent features shape: {train_latent_features.shape}")
    logger.info(f"Concatenated testing latent features shape: {test_latent_features.shape}")

    # Create and compile the classifier model
    classifier = models.create_classifier(
        input_dim=train_latent_features.shape[1], 
        num_classes=config['num_classes']
    )
    classifier.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate_classify']),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    
    # Train the classifier
    classifier.fit(train_latent_features, labels_tr,
                   batch_size=config['batch_size'],
                   epochs=config['epochs_classify'],
                   shuffle=True,
                   validation_data=(test_latent_features, labels_te),
                   verbose=2) # Use verbose=2 for one line per epoch
    # Evaluate the final model
    logger.info("Evaluating the classifier on the test set...")
    predictions = np.argmax(classifier.predict(test_latent_features), axis=1)
    accuracy = accuracy_score(labels_te, predictions)
    f1 = f1_score(labels_te, predictions, average='weighted')
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")

def run(config):
    """
    Main function to run the complete training and evaluation pipeline.

    Args:
        config (dict): A dictionary of configuration parameters.
    """
    # Load data
    data_tr_list, data_te_list, labels_tr, labels_te = features.load_and_preprocess_data(
        config['data_path'], config['view_list']
    )
    if data_tr_list is None:
        return

    # Pre-train encoders on each omics view
    encoders = pretrain_encoders(data_tr_list, config)
    
    # Train and evaluate the final classifier on concatenated features
    train_classifier(encoders, data_tr_list, data_te_list, labels_tr, labels_te, config)
