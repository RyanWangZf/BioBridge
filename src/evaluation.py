import pdb
import os
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr, pearsonr

def pairwise_similarity(emb, metric="manhattan"):
    """Compute the pairwise similarity between embeddings.

    Args:
        embs (np.ndarray): Embeddings of shape (n_samples, emb_dim).
        metric (str, optional): Metric to compute the pairwise distance in {"manhattan", "euclidean", "cosine"}. Defaults to "manhattan".

    Returns:
        np.ndarray: Pairwise similarity matrix of shape (n_samples * (n_samples - 1) / 2, ).
    """
    assert metric in ["manhattan", "euclidean", "cosine"], "metric must be one of 'manhattan', 'euclidean', 'cosine'"
    if metric == "manhattan":
        emb_norm = np.linalg.norm(emb,1,1)
    elif metric == "euclidean":
        emb_norm = np.linalg.norm(emb,2,1)
    else:
        emb_norm = None
    dist = pairwise_distances(emb, metric=metric)
    if emb_norm is not None:
        dist = dist / (emb_norm[None,:] + emb_norm[:,None])
    dist = 1 - dist
    dist = dist[np.triu_indices(dist.shape[0], k=1)]
    return dist

def semantic_similarity_inference(sim, gt):
    """Compute the spearman correlation between the predicted similarity and the groundtruth similarity.

    Args:
        sim (np.ndarray): Predicted similarity matrix of shape (n_samples * (n_samples - 1) / 2, ).
        gt (np.ndarray): Groundtruth similarity matrix of shape (n_samples, n_samples).
    """
    gt = gt[np.triu_indices(gt.shape[0], k=1)]
    corr = spearmanr(gt, sim).correlation
    return corr
