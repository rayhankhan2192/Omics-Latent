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