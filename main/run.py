import os
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Training Module")

from . import train

# Configuration for the BRCA Dataset
BRCA_CONFIG = {
    "data_path": os.path.join("data", "BRCA"),
    "view_list": [1, 2, 3],
    "num_classes": 5,
    "latent_dim": 128,      
    
    # Training Hyperparameters 
    "learning_rate_pretrain": 1e-3,
    "learning_rate_classify": 5e-4,
    "batch_size": 128,
    "epochs_pretrain": 100,     # More epochs for autoencoders to learn good features
    "epochs_classify": 150,     # Fewer epochs for the final classifier
}

#Configuration for the ROSMAP Dataset ---
ROSMAP_CONFIG = {
    "data_path": os.path.join("data", "ROSMAP"),
    "view_list": [1, 2, 3],
    "num_classes": 2,
    "latent_dim": 64,          # Smaller latent dim might be suitable
    
    # Training Hyperparameters ---
    "learning_rate_pretrain": 1e-3,
    "learning_rate_classify": 5e-4,
    "batch_size": 64,
    "epochs_pretrain": 150,
    "epochs_classify": 200,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a model for multi-omics data integration.')
    parser.add_argument('--dataset', 
                        type=str, 
                        required=True, 
                        choices=['BRCA', 'ROSMAP'],
                        help='The dataset to use for the experiment (choose between "BRCA" or "ROSMAP").')
    args = parser.parse_args()
    if args.dataset == 'BRCA':
        config_to_run = BRCA_CONFIG
    elif args.dataset == 'ROSMAP':
        config_to_run = ROSMAP_CONFIG
    else:
        raise ValueError("Invalid dataset choice. Please choose either 'BRCA' or 'ROSMAP'.")
    
    logger.info(f"Using configuration for {args.dataset} dataset.")
    train.run(config_to_run)

    logger.info("Training and evaluation complete.")