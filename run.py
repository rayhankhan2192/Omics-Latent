# import os
# import argparse
# import logging

# logging.basicConfig(level=logging.INFO, format='INFO:%(module)s:%(message)s')
# logger = logging.getLogger("Main Runner")

# from main import train

# BRCA_CONFIG = {
#     "data_path": os.path.join("Dataset", "BRCA"),
#     "view_list": [1, 2, 3],
#     "num_classes": 5,
#     "latent_dim": 128,
    
#     # Default Training Hyperparameters
#     "learning_rate_pretrain": 1e-3,
#     "learning_rate_classify": 5e-4,
#     "batch_size": 64,
#     "epochs_pretrain": 100,
#     "epochs_classify": 150,
# }

# # Configuration for the ROSMAP Dataset
# ROSMAP_CONFIG = {
#     "data_path": os.path.join("Dataset", "ROSMAP"),
#     "view_list": [1, 2, 3],
#     "num_classes": 2,
#     "latent_dim": 64,
    
#     # Default Training Hyperparameters
#     "learning_rate_pretrain": 1e-3,
#     "learning_rate_classify": 5e-4,
#     "batch_size": 64,
#     "epochs_pretrain": 150,
#     "epochs_classify": 200,
# }


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Train and evaluate a model for multi-omics data integration.')
    
#     parser.add_argument('--dataset', 
#                         type=str, 
#                         required=True, 
#                         choices=['BRCA', 'ROSMAP'],
#                         help='The dataset to use for the experiment (choose between "BRCA" or "ROSMAP").')
                        
#     parser.add_argument('--batch-size', 
#                         type=int, 
#                         default=None,
#                         help='Override the default batch size for training.')
                        
#     parser.add_argument('--epochs-pretrain', 
#                         type=int, 
#                         default=None,
#                         help='Override the default number of epochs for pre-training autoencoders.')

#     parser.add_argument('--epochs-classify', 
#                         type=int, 
#                         default=None,
#                         help='Override the default number of epochs for training the final classifier.')

#     parser.add_argument('--save-features', 
#                         action='store_true', 
#                         help='Save the latent and reconstructed features to CSV files.')

#     args = parser.parse_args()
    
#     if args.dataset == 'BRCA':
#         config_to_run = BRCA_CONFIG
#     elif args.dataset == 'ROSMAP':
#         config_to_run = ROSMAP_CONFIG
    
#     if args.batch_size is not None:
#         config_to_run['batch_size'] = args.batch_size
#         logger.info(f"Overriding batch size with command-line value: {args.batch_size}")

#     if args.epochs_pretrain is not None:
#         config_to_run['epochs_pretrain'] = args.epochs_pretrain
#         logger.info(f"Overriding pre-train epochs with command-line value: {args.epochs_pretrain}")

#     if args.epochs_classify is not None:
#         config_to_run['epochs_classify'] = args.epochs_classify
#         logger.info(f"Overriding classifier epochs with command-line value: {args.epochs_classify}")
    
#     config_to_run['save_features'] = args.save_features

#     logger.info(f"Starting experiment for {args.dataset} dataset.")
#     train.run(config_to_run)

#     logger.info("Experiment finished.")


import os
import argparse
import logging

# --- Setup Logging ---
# Use a specific logger name for clarity
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='INFO:%(name)s:%(message)s')


# --- Import the main training function ---
# Assuming 'main' is a package or directory containing train.py
from main import train

# === CONFIGURATION DICTIONARIES ===
# Denoising and Sparsity are OFF by default (set to None).
# To turn them on, set a value via the command line.

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
    "epochs_pretrain": 100,
    "epochs_classify": 150,

    # --- [NEW] Advanced AE/VAE Settings (Off by default) ---
    "denoising_noise_factor": None, # e.g., 0.2 (20% noise)
    "sparsity_l1_reg": None,        # e.g., 1e-5 (L1 penalty)
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

    # --- [NEW] Advanced AE/VAE Settings (Off by default) ---
    "denoising_noise_factor": None, # e.g., 0.2 (20% noise)
    "sparsity_l1_reg": None,        # e.g., 1e-5 (L1 penalty)
}


if __name__ == "__main__":
    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(description='Train and evaluate a model for multi-omics data integration.')
    
    # --- Required Argument ---
    parser.add_argument('--dataset', 
                        type=str, 
                        required=True, 
                        choices=['BRCA', 'ROSMAP'],
                        help='The dataset to use for the experiment (choose between "BRCA" or "ROSMAP").')
                        
    # --- Optional, Overriding Arguments ---
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

    parser.add_argument('--save-features', 
                        action='store_true', 
                        help='Save the latent and reconstructed features to CSV files.')

    # --- [NEW] Arguments for advanced models ---
    parser.add_argument('--denoising-noise',
                        type=float,
                        default=None,
                        help='Turn on Denoising Autoencoder with this noise factor (e.g., 0.2 for 20% dropout noise).')
    
    parser.add_argument('--sparsity-l1',
                        type=float,
                        default=None,
                        help='Turn on Sparse Autoencoder with this L1 regularization penalty (e.g., 1e-5).')

    args = parser.parse_args()
    
    # --- Select the base configuration based on the dataset ---
    if args.dataset == 'BRCA':
        config_to_run = BRCA_CONFIG
    elif args.dataset == 'ROSMAP':
        config_to_run = ROSMAP_CONFIG
    
    # --- Override config with command-line arguments if they were provided ---
    if args.batch_size is not None:
        config_to_run['batch_size'] = args.batch_size
        logger.info(f"Overriding batch size with command-line value: {args.batch_size}")

    if args.epochs_pretrain is not None:
        config_to_run['epochs_pretrain'] = args.epochs_pretrain
        logger.info(f"Overriding pre-train epochs with command-line value: {args.epochs_pretrain}")

    if args.epochs_classify is not None:
        config_to_run['epochs_classify'] = args.epochs_classify
        logger.info(f"Overriding classifier epochs with command-line value: {args.epochs_classify}")
    
    # --- [NEW] Add advanced settings to the config ---
    if args.denoising_noise is not None:
        config_to_run['denoising_noise_factor'] = args.denoising_noise
        logger.info(f"Enabling Denoising Autoencoder with noise factor: {args.denoising_noise}")
        
    if args.sparsity_l1 is not None:
        config_to_run['sparsity_l1_reg'] = args.sparsity_l1
        logger.info(f"Enabling Sparse Autoencoder with L1 penalty: {args.sparsity_l1}")
        
    config_to_run['save_features'] = args.save_features
    
    # --- Run the training process ---
    logger.info(f"Starting experiment for {args.dataset} dataset.")
    train.run(config_to_run)

    logger.info("Experiment finished.")

