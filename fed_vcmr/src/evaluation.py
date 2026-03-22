import numpy as np
from typing import List

def compute_metrics(ranks: np.ndarray):
    """Compute R@1, R@5, R@10, median rank, and mean rank."""
    metrics = {
        "R1": 100.0 * np.sum(ranks < 1) / len(ranks),
        "R5": 100.0 * np.sum(ranks < 5) / len(ranks),
        "R10": 100.0 * np.sum(ranks < 10) / len(ranks),
        "MedR": np.median(ranks) + 1,
        "MeanR": np.mean(ranks) + 1
    }
    return metrics

def get_ranks(similarity_matrix: np.ndarray):
    """
    similarity_matrix: (num_queries, num_videos)
    Assumes query i matches video i (diagonal is ground truth).
    """
    num_queries = similarity_matrix.shape[0]
    ranks = np.zeros(num_queries)
    for i in range(num_queries):
        sims = similarity_matrix[i]
        # Sort descending
        inds = np.argsort(-sims)
        # Find where the ground truth video is
        rank = np.where(inds == i)[0][0]
        ranks[i] = rank
    return ranks
