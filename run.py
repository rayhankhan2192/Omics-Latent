# import os
# import argparse
# import logging

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='INFO:%(name)s:%(message)s')

# from main import train

# BRCA_CONFIG = {
#     "data_path": os.path.join("Dataset", "BRCA"),
#     "view_list": [1, 2, 3],
#     "num_classes": 5,
#     "latent_dim": 96,
    
#     # Default Training Hyperparameters ---
#     "learning_rate_pretrain": 1e-3,
#     "learning_rate_classify": 5e-4,
#     "batch_size": 64,
#     "epochs_pretrain": 100,
#     "epochs_classify": 100,
#     "denoising_noise_factor": None, # e.g., 0.2 (20% noise)
#     "sparsity_l1_reg": None,        # e.g., 1e-5 (L1 penalty)
# }
# ROSMAP_CONFIG = {
#     "data_path": os.path.join("Dataset", "ROSMAP"),
#     "view_list": [1, 2, 3],
#     "num_classes": 2,
#     "latent_dim": 50,
    
#     # Default Training Hyperparameters ---
#     "learning_rate_pretrain": 1e-3,
#     "learning_rate_classify": 5e-4,
#     "batch_size": 64,
#     "epochs_pretrain": 100,
#     "epochs_classify": 100,

#     # --- [NEW] Advanced AE/VAE Settings (Off by default) ---
#     "denoising_noise_factor": None, # e.g., 0.2 (20% noise)
#     "sparsity_l1_reg": None,        # e.g., 1e-5 (L1 penalty)
# }


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Train and evaluate a model for multi-omics data integration.')
    
#     parser.add_argument('--dataset', type=str, required=True, choices=['BRCA', 'ROSMAP'],help='The dataset to use for the experiment (choose between "BRCA" or "ROSMAP").')
#     parser.add_argument('--batch-size', type=int, default=None,help='Override the default batch size for training.')    
#     parser.add_argument('--epochs-pretrain', type=int, default=None,help='Override the default number of epochs for pre-training autoencoders.')
#     parser.add_argument('--epochs-classify', type=int, default=None,help='Override the default number of epochs for training the final classifier.')
#     parser.add_argument('--save-features', action='store_true', help='Save the latent and reconstructed features to CSV files.')
#     parser.add_argument('--denoising-noise',type=float,default=None,help='Turn on Denoising Autoencoder with this noise factor (e.g., 0.2 for 20% dropout noise).')
#     parser.add_argument('--sparsity-l1',type=float,default=None,help='Turn on Sparse Autoencoder with this L1 regularization penalty (e.g., 1e-5).')

#     args = parser.parse_args()
    
#     # Select the base configuration based on the dataset ---
#     if args.dataset == 'BRCA':
#         config_to_run = BRCA_CONFIG
#     elif args.dataset == 'ROSMAP':
#         config_to_run = ROSMAP_CONFIG
    
#     # Override config with command-line arguments if they were provided ---
#     if args.batch_size is not None:
#         config_to_run['batch_size'] = args.batch_size
#         logger.info(f"Overriding batch size with command-line value: {args.batch_size}")

#     if args.epochs_pretrain is not None:
#         config_to_run['epochs_pretrain'] = args.epochs_pretrain
#         logger.info(f"Overriding pre-train epochs with command-line value: {args.epochs_pretrain}")

#     if args.epochs_classify is not None:
#         config_to_run['epochs_classify'] = args.epochs_classify
#         logger.info(f"Overriding classifier epochs with command-line value: {args.epochs_classify}")
    
#     # Add advanced settings to the config ---
#     if args.denoising_noise is not None:
#         config_to_run['denoising_noise_factor'] = args.denoising_noise
#         logger.info(f"Enabling Denoising Autoencoder with noise factor: {args.denoising_noise}")
        
#     if args.sparsity_l1 is not None:
#         config_to_run['sparsity_l1_reg'] = args.sparsity_l1
#         logger.info(f"Enabling Sparse Autoencoder with L1 penalty: {args.sparsity_l1}")
        
#     config_to_run['save_features'] = args.save_features
    
#     # Run the training process ---
#     logger.info(f"Starting experiment for {args.dataset} dataset.")
#     train.run(config_to_run)

#     logger.info("Experiment finished.")



# run.py
import os
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='INFO:%(name)s:%(message)s')

# Import the main training pipeline
from main import train

BRCA_CONFIG = {
    "data_path": os.path.join("Dataset", "BRCA"),
    "view_list": [1, 2, 3],
    "num_classes": 5,
    "latent_dim": 128, # You mentioned 128, so I've updated this
    
    # Default Training Hyperparameters ---
    "learning_rate_pretrain": 1e-3,
    "learning_rate_classify": 5e-4,
    "batch_size": 64,
    "epochs_pretrain": 100,
    "epochs_classify": 100,
    "denoising_noise_factor": None,
    "sparsity_l1_reg": None,
}
ROSMAP_CONFIG = {
    "data_path": os.path.join("Dataset", "ROSMAP"),
    "view_list": [1, 2, 3],
    "num_classes": 2,
    "latent_dim": 50,
    
    "learning_rate_pretrain": 1e-3,
    "learning_rate_classify": 5e-4,
    "batch_size": 64,
    "epochs_pretrain": 100,
    "epochs_classify": 100,
    "denoising_noise_factor": None,
    "sparsity_l1_reg": None,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multi-omics classification experiments.')
    
    # --- Primary Arguments ---
    parser.add_argument('--dataset', type=str, required=True, choices=['BRCA', 'ROSMAP'],
                        help='The dataset to use (BRCA or ROSMAP).')
    
    # parser.add_argument('--model-type', type=str, required=True, 
    #                     choices=['general', 'view_stacking', 'model_stacking', 'end_to_end'],
    #                     help='The architectural scenario to run.')
    
    parser.add_argument('--classifier', type=str, default='attention',
                        help='For "general" model type: which classifier to use (e.g., attention, graph, mlp).')

    # --- Hyperparameter Overrides ---
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override the default batch size.')    
    parser.add_argument('--epochs-pretrain', type=int, default=None,
                        help='Override pre-training epochs.')
    parser.add_argument('--epochs-classify', type=int, default=None,
                        help='Override classifier training epochs.')
    
    # --- Other Options ---
    parser.add_argument('--save-features', action='store_true', 
                        help='Save latent/reconstructed features to CSV.')
    parser.add_argument('--denoising-noise', type=float, default=None,
                        help='Turn on Denoising AE with this noise factor (e.g., 0.2).')
    parser.add_argument('--sparsity-l1', type=float, default=None,
                        help='Turn on Sparse AE with this L1 penalty (e.g., 1e-5).')
    
    parser.add_argument('--model-type', type=str, required=True, 
                        choices=['general', 'view_stacking', 'model_stacking', 'end_to_end', 'hybrid_fusion'], # <-- ADDED
                        help='The architectural scenario to run.')

    args = parser.parse_args()
    
    # --- 1. Select Base Config ---
    if args.dataset == 'BRCA':
        config_to_run = BRCA_CONFIG
    elif args.dataset == 'ROSMAP':
        config_to_run = ROSMAP_CONFIG
    
    # --- 2. Add Experiment-Specific Config ---
    config_to_run['model_type'] = args.model_type
    config_to_run['classifier'] = args.classifier
    config_to_run['save_features'] = args.save_features
    
    # --- 3. Override Hyperparameters ---
    if args.batch_size is not None:
        config_to_run['batch_size'] = args.batch_size
    if args.epochs_pretrain is not None:
        config_to_run['epochs_pretrain'] = args.epochs_pretrain
    if args.epochs_classify is not None:
        config_to_run['epochs_classify'] = args.epochs_classify
    if args.denoising_noise is not None:
        config_to_run['denoising_noise_factor'] = args.denoising_noise
    if args.sparsity_l1 is not None:
        config_to_run['sparsity_l1_reg'] = args.sparsity_l1
        
    # --- 4. Run the Training Pipeline ---
    logger.info(f"Starting experiment for {args.dataset} dataset.")
    logger.info(f"Running scenario: {args.model_type}")
    if args.model_type == 'general':
        logger.info(f"Using classifier: {args.classifier}")
        
    train.run_experiment(config_to_run)

    logger.info("Experiment finished.")