"""
Efficient IoU computation and Hungarian matching utilities.

Provides vectorized IoU matrix computation, optimal 1-to-1 Hungarian matching,
and IoU-based F-beta reward computation for bounding box predictions.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.optimize import linear_sum_assignment


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes (vectorized).

    Args:
        boxes1: (N, 4) array of boxes [x1, y1, x2, y2]
        boxes2: (M, 4) array of boxes [x1, y1, x2, y2]

    Returns:
        (N, M) IoU matrix where iou[i, j] = IoU(boxes1[i], boxes2[j])
    """
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    b1 = boxes1[:, np.newaxis, :]  # (N, 1, 4)
    b2 = boxes2[np.newaxis, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = np.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = np.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = np.minimum(b1[..., 3], b2[..., 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, np.newaxis] + area2[np.newaxis, :] - inter_area

    iou = np.where(union > 0, inter_area / union, 0.0)
    return iou.astype(np.float32)


def hungarian_matching(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    gt_labels: Optional[np.ndarray] = None,
    pred_labels: Optional[np.ndarray] = None,
    min_iou: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimal 1-to-1 matching between GT and predicted boxes using the
    Hungarian algorithm.

    Each GT box matches at most one predicted box and vice versa, preventing
    the issue where multiple predictions can "claim credit" for the same GT.

    Args:
        gt_boxes: (N, 4) ground truth boxes [x1, y1, x2, y2]
        pred_boxes: (M, 4) predicted boxes [x1, y1, x2, y2]
        gt_labels: (N,) optional ground truth labels for label-aware matching
        pred_labels: (M,) optional predicted labels
        min_iou: minimum IoU threshold for valid matches (default 0.0)

    Returns:
        matched_gt_indices: indices of matched GT boxes
        matched_pred_indices: indices of matched pred boxes (same length)
        matched_ious: IoU values for each matched pair
    """
    n_gt = len(gt_boxes)
    m_pred = len(pred_boxes)

    if n_gt == 0 or m_pred == 0:
        return (np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32))

    iou_matrix = compute_iou_matrix(gt_boxes, pred_boxes)

    if gt_labels is not None and pred_labels is not None:
        label_match = gt_labels[:, np.newaxis] == pred_labels[np.newaxis, :]
        iou_matrix = np.where(label_match, iou_matrix, 0.0)

    if min_iou > 0.0:
        iou_matrix = np.where(iou_matrix >= min_iou, iou_matrix, 0.0)

    cost_matrix = -iou_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matched_ious = iou_matrix[row_indices, col_indices]

    valid_mask = matched_ious > 0
    matched_gt_indices = row_indices[valid_mask].astype(np.int32)
    matched_pred_indices = col_indices[valid_mask].astype(np.int32)
    matched_ious = matched_ious[valid_mask].astype(np.float32)

    return matched_gt_indices, matched_pred_indices, matched_ious


def compute_hungarian_iou_f1_reward(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    gt_labels: Optional[np.ndarray] = None,
    pred_labels: Optional[np.ndarray] = None,
    eps: float = 1e-8,
    min_iou: float = 0.0,
    beta: float = 1.0,
) -> float:
    """
    Compute IoU F-beta reward using Hungarian 1-to-1 matching.

    Unlike greedy matching, Hungarian matching ensures each GT matches at most
    one prediction and vice versa, properly penalizing redundant predictions.

        Recall = sum(matched_ious) / n_gt
        Precision = sum(matched_ious) / m_pred
        F_beta = (1 + beta^2) * P * R / (beta^2 * P + R + eps)

    Args:
        gt_boxes: (N, 4) ground truth boxes
        pred_boxes: (M, 4) predicted boxes
        gt_labels: (N,) ground truth labels (optional)
        pred_labels: (M,) predicted labels (optional)
        eps: small constant to prevent division by zero
        min_iou: minimum IoU threshold for matching
        beta: beta for F-beta score (default 1.0 = F1)

    Returns:
        F-beta IoU reward in [0, 1]
    """
    n_gt = len(gt_boxes)
    m_pred = len(pred_boxes)

    if n_gt == 0 and m_pred == 0:
        return 1.0
    if n_gt == 0 or m_pred == 0:
        return 0.0

    matched_gt_idx, matched_pred_idx, matched_ious = hungarian_matching(
        gt_boxes, pred_boxes, gt_labels, pred_labels, min_iou
    )

    recall = float(np.sum(matched_ious)) / n_gt
    precision = float(np.sum(matched_ious)) / m_pred

    beta_sq = beta * beta
    f_beta = (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall + eps)
    return float(f_beta)
