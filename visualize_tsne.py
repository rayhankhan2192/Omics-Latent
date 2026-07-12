import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

# Re-use your existing multi-view pipeline layers
from main import features
from main import train

def run_dataset_tsne_pipeline():
    parser = argparse.ArgumentParser(description="Generate publication-quality t-SNE projections for Multi-Omics spaces.")
    parser.add_argument('--dataset', type=str, required=True, choices=['BRCA', 'ROSMAP'],
                        help='The target dataset cohort to visualize.')
    args = parser.parse_args()
    
    # Define exact structural configuration maps matching run.py presets
    CONFIGS = {
        "BRCA": {
            "data_path": os.path.join("Dataset", "BRCA"),
            "view_list": [1, 2, 3],
            "num_classes": 5,
            "latent_dim": 128,
            "learning_rate_pretrain": 1e-3,
            "batch_size": 64,
            "epochs_pretrain": 100,
            "class_names": ['Normal-like', 'Basal-like', 'HER2-enriched', 'Luminal A', 'Luminal B']
        },
        "ROSMAP": {
            "data_path": os.path.join("Dataset", "ROSMAP"),
            "view_list": [1, 2, 3],
            "num_classes": 2,
            "latent_dim": 50,
            "learning_rate_pretrain": 1e-3,
            "batch_size": 64,
            "epochs_pretrain": 100,
            "class_names": ['Control (Class 0)', 'AD Case (Class 1)']
        }
    }
    
    target_config = CONFIGS[args.dataset]
    print(f"--- Starting t-SNE Extraction Pipeline for Cohort: {args.dataset} ---")
    
    # 1. Load aligned patient metrics via features extraction engine
    data_tr_list, data_te_list, labels_tr, labels_te = features.load_and_preprocess_data(
        target_config['data_path'], target_config['view_list']
    )
    
    le = LabelEncoder()
    labels_te_encoded = le.fit_transform(labels_te)
    
    # 2. Extract regularized phase-1 latent dimensions
    print("Extracting pre-trained latent manifolds...")
    encoders, _ = train.pretrain_encoders(data_tr_list, target_config)
    test_latent_list = []
    for i, encoder in enumerate(encoders):
        test_latent = encoder.predict(data_te_list[i])
        test_latent_list.append(test_latent)
        
    # 3. Construct the distinct operational feature layouts
    raw_space = np.concatenate(data_te_list, axis=1)
    latent_space = np.concatenate(test_latent_list, axis=1)
    hybrid_space = np.concatenate([raw_space, latent_space], axis=1)
    
    spaces = [
        {"name": "Raw Feature Space", "data": raw_space},
        {"name": "Pure Latent Space", "data": latent_space},
        {"name": "Proposed Hybrid Space", "data": hybrid_space}
    ]
    
    # 4. Generate spatial comparison plots using matplotlib subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.set_theme(style="whitegrid")
    
    # Use distinct palettes depending on classification boundary targets
    active_cmap = "coolwarm" if args.dataset == "ROSMAP" else "tab10"
    
    for idx, space in enumerate(spaces):
        print(f"Computing 2D projections for space: {space['name']}...")
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        embeddings = tsne.fit_transform(space['data'])
        
        ax = axes[idx]
        scatter = ax.scatter(
            embeddings[:, 0], embeddings[:, 1], 
            c=labels_te_encoded, cmap=active_cmap, alpha=0.8, edgecolors='w', s=55
        )
        ax.set_title(space['name'], fontsize=14, fontweight='bold')
        ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
        
    # Append localized legend layout mapping identical group variables
    handles, _ = scatter.legend_elements()
    fig.legend(handles, target_config['class_names'], loc='lower center', ncol=len(target_config['class_names']), bbox_to_anchor=(0.5, -0.06), fontsize=12)
    plt.tight_layout()
    
    output_filename = f"S5_Fig_{args.dataset}_tsne_manifolds.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Manifold comparison figure successfully exported to: {output_filename}\n")
    plt.close()

if __name__ == "__main__":
    run_dataset_tsne_pipeline()