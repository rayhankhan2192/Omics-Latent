import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def generate_tsne_plots(raw_data, encoder_models, hybrid_classifier, labels, class_names, file_prefix="brca_"):
    """
    Extracts features from the raw, latent, and hybrid spaces,
    projects them using t-SNE, and saves a side-by-side comparison plot.
    """
    sns.set_theme(style="whitegrid")
    
    # --- 1. Extract Representations ---
    # Raw Space: Concatenated raw modalities
    # (Assuming raw_data is a list of arrays: [mRNA, Meth, miRNA])
    raw_space = np.concatenate(raw_data, axis=1)
    
    # Latent Space: Concatenated encoder outputs
    latent_views = [encoder.predict(view) for encoder, view in zip(encoder_models, raw_data)]
    latent_space = np.concatenate(latent_views, axis=1)
    
    # Hybrid Space: Extract right before the final classification head
    # We create a feature extractor model targeting the concatenation layer
    concat_layer_name = "hybrid_concat_layer" # Update this to match your actual layer name in models.py
    feature_extractor = Model(inputs=hybrid_classifier.input, 
                              outputs=hybrid_classifier.get_layer(concat_layer_name).output)
    hybrid_space = feature_extractor.predict(raw_data)
    
    spaces = [raw_space, latent_space, hybrid_space]
    titles = ["(A) Raw Feature Space", "(B) Pure Latent Space", "(C) Proposed Hybrid Fusion Space"]
    
    # --- 2. Compute t-SNE and Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, (space, title) in enumerate(zip(spaces, titles)):
        # Run t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        embedded_features = tsne.fit_transform(space)
        
        # Plot each class group cleanly
        for class_idx, class_name in enumerate(class_names):
            mask = (labels == class_idx)
            axes[i].scatter(
                embedded_features[mask, 0], 
                embedded_features[mask, 1], 
                label=class_name, 
                alpha=0.75, 
                edgecolors='w', 
                s=50
            )
            
        axes[i].set_title(title, fontsize=14, fontweight='bold')
        axes[i].set_xlabel("t-SNE Dimension 1", fontsize=11)
        axes[i].set_ylabel("t-SNE Dimension 2", fontsize=11)
        axes[i].grid(True, linestyle='--', alpha=0.5)
        
    # Single unified legend outside the subplots
    handles, labels_legend = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
               ncol=len(class_names), fontsize=12, frameon=True)
    
    plt.tight_layout()
    output_filename = f"{file_prefix}tsne_comparison.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Beautiful t-SNE comparison plot saved as: {output_filename}")