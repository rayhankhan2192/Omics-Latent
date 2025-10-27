import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='INFO:%(module)s:%(message)s')
logger = logging.getLogger("Main Runner")

from main import train

BRCA_CONFIG = {
    "data_path": os.path.join("Dataset", "BRCA"),
    "view_list": [1, 2, 3],
    "num_classes": 5,
    "latent_dim": 128,
    
    # Default Training Hyperparameters
    "learning_rate_pretrain": 1e-3,
    "learning_rate_classify": 5e-4,
    "batch_size": 64,
    "epochs_pretrain": 100,
    "epochs_classify": 150,
}

# Configuration for the ROSMAP Dataset
ROSMAP_CONFIG = {
    "data_path": os.path.join("Dataset", "ROSMAP"),
    "view_list": [1, 2, 3],
    "num_classes": 2,
    "latent_dim": 64,
    
    # Default Training Hyperparameters
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
    
    config_to_run['save_features'] = args.save_features

    logger.info(f"Starting experiment for {args.dataset} dataset.")
    train.run(config_to_run)

    logger.info("Experiment finished.")