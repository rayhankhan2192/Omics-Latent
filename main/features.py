# import logging
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

# import os
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# logger = logging.getLogger("Features Module")

# def load_and_preprocess_data(data_path, view_list):
#     """
#     Loads, preprocesses, and scales multi-omics data from the specified path.

#     Args:
#         data_path (str): The path to the dataset folder (e.g., './data/BRCA').
#         view_list (list): A list of integers representing the omics views to load.

#     Returns:
#         tuple: A tuple containing lists of training and testing data for each view,
#                and the training and testing labels.
#     """
#     print(f"--- Loading data from {data_path} ---")
    
#     #load data from each view
#     data_tr_list = []
#     data_te_list = []
#     for view in view_list:
#         try:
#             tr_df = pd.read_csv(os.path.join(data_path, f'{view}_tr.csv'), header=None)
#             te_df = pd.read_csv(os.path.join(data_path, f'{view}_te.csv'), header=None)
#             data_tr_list.append(tr_df.values)
#             data_te_list.append(te_df.values)
#             logger.info(f"View {view} data loaded successfully.")
#             logger.info(f"Training data shape for view {view}: {tr_df.shape}")
#             logger.info(f"Testing data shape for view {view}: {te_df.shape}")
#         except FileNotFoundError:
#             logger.error(f"Training data file for view {view} not found.")
#             return None, None, None, None
#         except pd.errors.EmptyDataError:
#             logger.error(f"Training data file for view {view} is empty.")
#             return None, None, None, None
        
#         # Load Labels
#         try:
#             labels_tr = pd.read_csv(os.path.join(data_path, 'labels_tr.csv'), header=None).values.ravel()
#             labels_te = pd.read_csv(os.path.join(data_path, 'labels_te.csv'), header=None).values.ravel()
#         except FileNotFoundError:
#             logger.error("Label files not found.")
#             return None, None, None, None
        
#         # Standardize data (scale to zero mean and unit variance)
#         for i in range(len(data_tr_list)):
#             scaler = StandardScaler()
#             data_tr_list[i] = scaler.fit_transform(data_tr_list[i])
#             data_te_list[i] = scaler.transform(data_te_list[i])
#             logger.info(f"Data for view {view_list[i]} standardized.")

#             logger.info(f"Training data shape for view {view_list[i]} after scaling: {data_tr_list[i].shape}")
#             logger.info(f"Testing data shape for view {view_list[i]} after scaling: {data_te_list[i].shape}")

#             # Convert labels to be 0-indexed for TensorFlow
#             labels_tr = labels_tr - 1
#             labels_te = labels_te - 1
#             logger.info(f"Data loading and preprocessing complete")
#     return data_tr_list, data_te_list, labels_tr, labels_te


import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("Features Module")

def load_and_preprocess_data(data_path, view_list):
    """
    Loads, preprocesses, and scales multi-omics data from the specified path.
    """
    print(f"--- Loading data from {data_path} ---")
    
    data_tr_list = []
    data_te_list = []
    
    #STEP 1: Load all view data first ===
    for view in view_list:
        try:
            tr_df = pd.read_csv(os.path.join(data_path, f'{view}_tr.csv'), header=None)
            te_df = pd.read_csv(os.path.join(data_path, f'{view}_te.csv'), header=None)
            data_tr_list.append(tr_df.values)
            data_te_list.append(te_df.values)
            logger.info(f"View {view} data loaded. Train shape: {tr_df.shape}, Test shape: {te_df.shape}")
        except FileNotFoundError:
            logger.error(f"Data file for view {view} not found.")
            return None, None, None, None
        except pd.errors.EmptyDataError:
            logger.error(f"Data file for view {view} is empty.")
            return None, None, None, None

    # STEP 2: Load labels ONCE 
    try:
        labels_tr = pd.read_csv(os.path.join(data_path, 'labels_tr.csv'), header=None).values.ravel()
        labels_te = pd.read_csv(os.path.join(data_path, 'labels_te.csv'), header=None).values.ravel()
        logger.info(f"Labels loaded. Train shape: {labels_tr.shape}, Test shape: {labels_te.shape}")
    except FileNotFoundError:
        logger.error("Label files not found.")
        return None, None, None, None
        
    # STEP 3: Scale data ONCE (in a separate loop)
    for i in range(len(data_tr_list)):
        scaler = StandardScaler()
        data_tr_list[i] = scaler.fit_transform(data_tr_list[i])
        data_te_list[i] = scaler.transform(data_te_list[i])
        logger.info(f"Data for view {view_list[i]} standardized.")

    # STEP 4: Convert labels ONCE 
    # Convert labels to be 0-indexed for TensorFlow
    labels_tr = labels_tr - 1
    labels_te = labels_te - 1
    logger.info("Labels converted to 0-indexing.")
    
    logger.info(f"Data loading and preprocessing complete.")
    return data_tr_list, data_te_list, labels_tr, labels_te