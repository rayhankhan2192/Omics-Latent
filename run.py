import os
import argparse
import logging

# --- Setup Logging ---
# Using a logger is better than print for tracking experiments
logging.basicConfig(level=logging.INFO, format='INFO:%(module)s:%(message)s')
logger = logging.getLogger("Main Runner")

# --- Import the main training function ---
# Assumes train.py is in a sub-folder named 'main'
from main import train

# === CONFIGURATION DICTIONARIES ===
# These hold the default parameters for each dataset.
# They can be overridden by command-line arguments.

# --- Configuration for the BRCA Dataset ---
BRCA_CONFIG = {
    "data_path": os.path.join("DataSet", "BRCA"),
    "view_list": [1, 2, 3],
    "num_classes": 5,
    "latent_dim": 128,
    
    # --- Default Training Hyperparameters ---
    "learning_rate_pretrain": 1e-3,
    "learning_rate_classify": 5e-4,
    "batch_size": 64,
    "epochs_pretrain": 150,  # Default epochs for autoencoders
    "epochs_classify": 200,  # Default epochs for the final classifier
}

# --- Configuration for the ROSMAP Dataset ---
ROSMAP_CONFIG = {
    "data_path": os.path.join("DataSet", "ROSMAP"),
    "view_list": [1, 2, 3],
    "num_classes": 2,
    "latent_dim": 64,
    
    # --- Default Training Hyperparameters ---
    "learning_rate_pretrain": 1e-3,
    "learning_rate_classify": 5e-4,
    "batch_size": 64,
    "epochs_pretrain": 150,
    "epochs_classify": 200,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a model for multi-omics data integration.')
    
    # --- Required Argument ---
    parser.add_argument('--dataset', 
                        type=str, 
                        required=True, 
                        choices=['BRCA', 'ROSMAP'],
                        help='The dataset to use for the experiment (choose between "BRCA" or "ROSMAP").')
                        
    parser.add_argument('--batch-size', 
                        type=int, 
                        default=None,
                        help='Override the default batch size for training.')
                        
    parser.add_argument('--epochs-pretrain', 
                        type=int, 
                        default=None,
                        help='Override the default number of epochs for pre-training autoencoders.')

    parser.add_argument('--epochs-classify', 
                        type=int, 
                        default=None,
                        help='Override the default number of epochs for training the final classifier.')

    args = parser.parse_args()
    
    if args.dataset == 'BRCA':
        config_to_run = BRCA_CONFIG
    elif args.dataset == 'ROSMAP':
        config_to_run = ROSMAP_CONFIG
    
    if args.batch_size is not None:
        config_to_run['batch_size'] = args.batch_size
        logger.info(f"Overriding batch size with command-line value: {args.batch_size}")

    if args.epochs_pretrain is not None:
        config_to_run['epochs_pretrain'] = args.epochs_pretrain
        logger.info(f"Overriding pre-train epochs with command-line value: {args.epochs_pretrain}")

    if args.epochs_classify is not None:
        config_to_run['epochs_classify'] = args.epochs_classify
        logger.info(f"Overriding classifier epochs with command-line value: {args.epochs_classify}")
    logger.info(f"Starting experiment for {args.dataset} dataset.")
    train.run(config_to_run)

    logger.info("Experiment finished.")

