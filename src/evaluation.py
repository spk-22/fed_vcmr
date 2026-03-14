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

def compute_iou(pred_segment, gt_segment):
    """
    Compute Temporal Intersection over Union (IoU) between two segments.
    segments are tuples or lists: (start_time, end_time)
    """
    pred_start, pred_end = pred_segment
    gt_start, gt_end = gt_segment
    
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    intersection = max(0.0, intersection_end - intersection_start)
    
    pred_duration = max(0.0, pred_end - pred_start)
    gt_duration = max(0.0, gt_end - gt_start)
    
    union = pred_duration + gt_duration - intersection
    
    if union <= 0.0:
        return 0.0
        
    return intersection / union

def evaluate_moment_retrieval(predictions: List[tuple], ground_truths: List[tuple], iou_thresholds=[0.3, 0.5, 0.7]):
    """
    Evaluates moment retrieval based on IoU thresholds.
    predictions: list of predicted (start, end) segments, sorted by rank for a single query.
    ground_truths: list of ground truth (start, end) segments for the query.
    
    Returns a dictionary of boolean indicators for R@1, R@5 for each threshold.
    """
    results = {f"R1_IoU@{th}": False for th in iou_thresholds}
    for th in iou_thresholds:
        results[f"R5_IoU@{th}"] = False
        
    for k in [1, 5]:
        top_k_preds = predictions[:k]
        for pred in top_k_preds:
            # If pred matches ANY ground truth with IoU > th
            max_iou = 0.0
            for gt in ground_truths:
                iou = compute_iou(pred, gt)
                max_iou = max(max_iou, iou)
                
            for th in iou_thresholds:
                if max_iou >= th:
                    if k == 1:
                        results[f"R1_IoU@{th}"] = True
                    results[f"R5_IoU@{th}"] = True
                    
    return results
