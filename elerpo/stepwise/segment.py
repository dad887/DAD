"""
Data structures for EleRPO training segments.

A Segment ties a subset of predictions and ground-truth boxes together with
the token span that produced those predictions, enabling per-token advantage
assignment in the GRPO loss.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Segment:
    """
    A segment containing 0+ predictions and 0+ GT boxes.

    The reward is computed by comparing preds vs GTs in the segment.
    Token spans map segment rewards back to per-token advantages.

    Attributes:
        segment_id: Unique identifier for this segment.
        pred_boxes: (M, 4) predicted boxes [x1, y1, x2, y2].
        pred_labels: List of M predicted labels, or None.
        pred_indices: Indices of these predictions in the full prediction list.
        segment_token_span: Overall (start, end) token span, or None if no preds.
        gt_boxes: (N, 4) ground truth boxes.
        gt_labels: List of N GT labels, or None.
        gt_indices: Indices of these GTs in the full GT list.
        match_info: Dict with matching details (IoU, matched counts, etc.).
    """
    segment_id: int
    segment_name: str = ""

    pred_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    pred_labels: Optional[List[str]] = None
    pred_indices: List[int] = field(default_factory=list)
    pred_token_spans: List[Tuple[int, int]] = field(default_factory=list)
    segment_token_span: Optional[Tuple[int, int]] = None

    gt_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    gt_labels: Optional[List[str]] = None
    gt_indices: List[int] = field(default_factory=list)

    match_info: Optional[dict] = None

    @property
    def n_pred(self) -> int:
        return len(self.pred_boxes)

    @property
    def n_gt(self) -> int:
        return len(self.gt_boxes)

    @property
    def has_predictions(self) -> bool:
        return self.n_pred > 0

    def __repr__(self) -> str:
        return (f"Segment(id={self.segment_id}, name='{self.segment_name}', "
                f"n_pred={self.n_pred}, n_gt={self.n_gt}, "
                f"token_span={self.segment_token_span})")


@dataclass
class SlicingResult:
    """
    Complete result of slicing a sample into segments.

    Attributes:
        segments: List of Segment objects (one per predicted box).
        all_pred_boxes: (M, 4) all predicted boxes.
        all_pred_labels: List of all predicted labels, or None.
        all_pred_token_spans: Token spans for all predictions.
        all_gt_boxes: (N, 4) all ground truth boxes.
        all_gt_labels: List of all GT labels, or None.
    """
    segments: List[Segment]

    all_pred_boxes: np.ndarray
    all_pred_labels: Optional[List[str]]
    all_pred_token_spans: List[Tuple[int, int]]

    all_gt_boxes: np.ndarray
    all_gt_labels: Optional[List[str]]

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_total_pred(self) -> int:
        return len(self.all_pred_boxes)

    @property
    def n_total_gt(self) -> int:
        return len(self.all_gt_boxes)

    def __repr__(self) -> str:
        return (f"SlicingResult(n_segments={self.n_segments}, "
                f"n_pred={self.n_total_pred}, n_gt={self.n_total_gt})")
