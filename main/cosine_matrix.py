"""Cosine-similarity patient matrices and label-free neighbor aggregation.

For each view, the training-to-training similarity matrix is formed as
S = L2normalize(X_train) @ L2normalize(X_train).T.
For held-out patients S_query = L2normalize(X_query) @ L2normalize(X_train).T.
No test/validation samples are used as references, and labels are never used.
"""
import numpy as np
from sklearn.preprocessing import normalize


def cosine_matrix(query, reference):
    q = normalize(np.asarray(query, dtype=np.float32), axis=1)
    r = normalize(np.asarray(reference, dtype=np.float32), axis=1)
    return (q @ r.T).astype(np.float32)


def topk_adjacency(similarity, k=5, self_reference=False):
    """Top-k nonnegative cosine weights; rows sum to one when edges exist."""
    sim = np.asarray(similarity, dtype=np.float32).copy()
    if self_reference:
        if sim.shape[0] != sim.shape[1]:
            raise ValueError('Self-reference adjacency must be square')
        np.fill_diagonal(sim, -np.inf)
    sim = np.maximum(sim, 0.0)  # ignore negatively correlated patients
    if self_reference:
        np.fill_diagonal(sim, -np.inf)  # do not allow self even when ties are zero
    n_rows, n_cols = sim.shape
    if n_cols < 1 or (self_reference and n_cols < 2):
        raise ValueError('Not enough reference patients for cosine adjacency')
    k = min(int(k), n_cols - (1 if self_reference else 0))
    if k < 1:
        raise ValueError('k must be at least 1')
    # argsort gives stable, unambiguous top-k selection (small ROSMAP cohort)
    index = np.argsort(sim, axis=1)[:, -k:]
    weights = np.take_along_axis(sim, index, axis=1)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    row_sum = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, row_sum, out=np.zeros_like(weights), where=row_sum > 0)
    adjacency = np.zeros_like(sim)
    np.put_along_axis(adjacency, index, weights, axis=1)
    return adjacency


def cosine_hybrid_features(original_fit, original_query, latent_fit, k=5,
                           self_reference=False, return_matrices=False):
    """Return aggregated latent vectors obtained via the cosine matrix.

    original_fit: reference patients from classifier-fitting partition only.
    original_query: fit, validation, or test patients.
    latent_fit: learned latent representations of reference patients.
    """
    s = cosine_matrix(original_query, original_fit)
    a = topk_adjacency(s, k=k, self_reference=self_reference)
    neighbor_latent = a @ np.asarray(latent_fit, dtype=np.float32)
    if return_matrices:
        return neighbor_latent.astype(np.float32), s, a
    return neighbor_latent.astype(np.float32)
