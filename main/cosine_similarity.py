"""Train-reference cosine kNN feature augmentation for hybrid multi-omics inputs.

No labels are used; test patients only query the training reference patients.
Train queries exclude themselves to prevent trivial similarity=1 leakage.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def _unit_blocks(original, latent):
    """Give the original and latent blocks equal contribution to cosine distance."""
    a = normalize(np.asarray(original, dtype=np.float32), norm='l2')
    b = normalize(np.asarray(latent, dtype=np.float32), norm='l2')
    return normalize(np.concatenate((a, b), axis=1), norm='l2')


def _neighbour_features(indices, distances, reference_latent):
    # Angular cosine similarity: 1 - cosine distance, possibly negative.
    similarities = np.clip(1.0 - distances, -1.0, 1.0)
    # Stable nonnegative attention-like weights; all-negative neighborhoods still work.
    scores = np.exp((similarities - similarities.max(axis=1, keepdims=True)) / 0.2)
    weights = scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-8)
    neighbor_latent = reference_latent[indices]  # (N, k, latent_dim)
    aggregation = np.sum(weights[..., None] * neighbor_latent, axis=1)
    stats = np.stack([similarities.mean(axis=1), similarities.max(axis=1)], axis=1)
    return np.concatenate([aggregation, stats], axis=1).astype(np.float32)


def augment_hybrid_with_cosine(train_original, test_original,
                               train_latent, test_latent, k=5):
    """Return training and test hybrid features including cosine kNN summaries.

    Retrieval: cosine over separately L2-normalized original and latent blocks.
    Appended features: cosine-weighted mean of training neighbor latent features,
    mean cosine similarity, maximum cosine similarity.
    """
    n_train = len(train_original)
    if n_train < 2:
        raise ValueError('At least two training samples are required for cosine kNN.')
    if k < 1:
        raise ValueError('k must be >= 1')
    k = min(int(k), n_train - 1)
    train_repr = _unit_blocks(train_original, train_latent)
    test_repr = _unit_blocks(test_original, test_latent)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
    nbrs.fit(train_repr)
    # Query train with k+1 to remove exact self-index, including duplicate-feature cases.
    train_dist, train_ind = nbrs.kneighbors(train_repr, n_neighbors=k + 1)
    keep_ind = np.empty((n_train, k), dtype=np.int64)
    keep_dist = np.empty((n_train, k), dtype=np.float32)
    for i in range(n_train):
        valid = train_ind[i] != i
        keep_ind[i] = train_ind[i][valid][:k]
        keep_dist[i] = train_dist[i][valid][:k]
    test_dist, test_ind = nbrs.kneighbors(test_repr, n_neighbors=k)
    reference_latent = np.asarray(train_latent, dtype=np.float32)
    tr_neighbor = _neighbour_features(keep_ind, keep_dist, reference_latent)
    te_neighbor = _neighbour_features(test_ind, test_dist, reference_latent)
    tr_hybrid = np.concatenate([train_original, train_latent, tr_neighbor], axis=1).astype(np.float32)
    te_hybrid = np.concatenate([test_original, test_latent, te_neighbor], axis=1).astype(np.float32)
    return tr_hybrid, te_hybrid
