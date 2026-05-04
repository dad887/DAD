"""
Slicing strategies for EleRPO training.

A slicing strategy partitions model predictions and ground truth into
per-object segments, enabling fine-grained per-token advantage assignment.

The primary strategy is HungarianSlicing, which uses optimal 1-to-1 matching
between predicted and ground-truth bounding boxes.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .iou_utils import compute_iou_matrix
from .segment import Segment, SlicingResult


# Short label codes used in the semicolon_with_type output format
TYPE_SHORT_TO_FULL = {"t": "text", "v": "visual", "n": "annotation"}


def parse_semicolon_with_type(
    completion: str,
    tokenizer: Any,
    delimiter: str = ";",
    include_delimiter_in_span: bool = False,
) -> Tuple[np.ndarray, Optional[List[str]], List[Tuple[int, int]]]:
    """
    Parse a semicolon-delimited model completion into boxes, labels, and
    token spans.

    Expected format per element: ``x1,y1,x2,y2,t`` where the last field is a
    single-character type code (``t`` for text, ``v`` for visual).  Extra
    trailing fields (e.g. z-index) are silently ignored.

    Args:
        completion: Raw model output string.
        tokenizer: HuggingFace-compatible tokenizer (must support ``encode``).
        delimiter: Box delimiter (default ``";"``).
        include_delimiter_in_span: Extend each box's token span to include
            the trailing delimiter character.

    Returns:
        pred_boxes: (M, 4) float32 array of predicted boxes.
        pred_labels: List of M label strings, or None if parsing fails.
        token_spans: List of M ``(start, end)`` token index pairs.
    """
    completion = completion.strip()
    if not completion:
        return np.zeros((0, 4), dtype=np.float32), None, []

    boxes = []
    labels = []
    token_spans = []

    parts = completion.split(delimiter)
    current_char_pos = 0

    for part_idx, part in enumerate(parts):
        part_stripped = part.strip()
        if not part_stripped:
            current_char_pos += len(part) + (1 if part_idx < len(parts) - 1 else 0)
            continue

        coords = part_stripped.split(",")

        if len(coords) >= 5:
            try:
                box = [float(coords[i].strip()) for i in range(4)]
                label = TYPE_SHORT_TO_FULL.get(
                    coords[4].strip(), coords[4].strip()
                )

                start_char = completion.find(part_stripped, current_char_pos)
                end_char = start_char + len(part_stripped)
                prefix = completion[:start_char]
                prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
                token_start = len(prefix_tokens)

                if (include_delimiter_in_span
                        and end_char < len(completion)
                        and completion[end_char] == delimiter):
                    span_text = completion[start_char:end_char + 1]
                else:
                    span_text = part_stripped
                span_tokens = tokenizer.encode(span_text, add_special_tokens=False)
                token_end = token_start + len(span_tokens)

                boxes.append(box)
                labels.append(label)
                token_spans.append((token_start, token_end))

            except (ValueError, IndexError):
                pass

        current_char_pos += len(part) + (1 if part_idx < len(parts) - 1 else 0)

    pred_boxes = (np.array(boxes, dtype=np.float32).reshape(-1, 4)
                  if boxes else np.zeros((0, 4), dtype=np.float32))
    pred_labels = labels if labels else None

    return pred_boxes, pred_labels, token_spans


class HungarianSlicing:
    """
    Slicing strategy using optimal 1-to-1 Hungarian matching.

    Each predicted box becomes its own segment, matched to at most one GT via
    the Hungarian algorithm based on IoU.

    Behavior:
      - Extra predictions (FP): segment with 0 GTs, reward = 0.
      - Extra GTs (FN): not assigned to any segment, but tracked in
        ``match_info`` so the coverage penalty can scale all rewards by
        ``n_matched / n_gt``.

    Args:
        iou_threshold: Minimum IoU to consider a match valid.
        include_delimiter_in_span: Extend token spans to include the trailing
            delimiter character.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        include_delimiter_in_span: bool = False,
    ):
        self.iou_threshold = iou_threshold
        self.include_delimiter_in_span = include_delimiter_in_span

    def slice(
        self,
        completion: str,
        gt_boxes: np.ndarray,
        gt_labels: Optional[List[str]],
        tokenizer: Any,
    ) -> SlicingResult:
        """
        Parse ``completion``, match predictions to ``gt_boxes`` via the
        Hungarian algorithm, and return one Segment per predicted box.

        Args:
            completion: Model output string in ``semicolon_with_type`` format.
            gt_boxes: (N, 4) ground truth boxes.
            gt_labels: List of N GT labels, or None.
            tokenizer: HuggingFace-compatible tokenizer.

        Returns:
            SlicingResult containing all per-box segments.
        """
        pred_boxes, pred_labels, token_spans = parse_semicolon_with_type(
            completion, tokenizer,
            include_delimiter_in_span=self.include_delimiter_in_span,
        )

        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes) if gt_boxes is not None else 0
        if gt_boxes is None:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)

        # --- Hungarian matching ---
        matched_pairs = []
        matched_gt_set = set()

        if n_pred > 0 and n_gt > 0:
            iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
            iou_matrix = np.nan_to_num(iou_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(iou_matrix, 0.0, 1.0, out=iou_matrix)
            cost_matrix = 1 - iou_matrix

            pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

            for pi, gi in zip(pred_indices, gt_indices):
                iou = iou_matrix[pi, gi]
                if iou >= self.iou_threshold:
                    matched_pairs.append((int(pi), int(gi), float(iou)))
                    matched_gt_set.add(int(gi))

        matched_pred_to_gt = {p: (g, iou) for p, g, iou in matched_pairs}
        n_matched = len(matched_pairs)
        sum_iou_total = sum(iou for _, _, iou in matched_pairs)

        # --- Build one Segment per prediction ---
        segments = []
        for pi in range(n_pred):
            if pi in matched_pred_to_gt:
                gi, iou = matched_pred_to_gt[pi]
                seg = Segment(
                    segment_id=pi,
                    segment_name=f"pred_{pi}_matched",
                    pred_boxes=pred_boxes[pi:pi+1],
                    pred_labels=[pred_labels[pi]] if pred_labels else None,
                    pred_indices=[pi],
                    pred_token_spans=[token_spans[pi]],
                    segment_token_span=token_spans[pi],
                    gt_boxes=gt_boxes[gi:gi+1],
                    gt_labels=[gt_labels[gi]] if gt_labels else None,
                    gt_indices=[gi],
                    match_info={
                        "matched": True,
                        "iou": iou,
                        "n_matched_total": n_matched,
                        "n_gt_total": n_gt,
                        "sum_iou_total": sum_iou_total,
                    },
                )
            else:
                seg = Segment(
                    segment_id=pi,
                    segment_name=f"pred_{pi}_unmatched",
                    pred_boxes=pred_boxes[pi:pi+1],
                    pred_labels=[pred_labels[pi]] if pred_labels else None,
                    pred_indices=[pi],
                    pred_token_spans=[token_spans[pi]],
                    segment_token_span=token_spans[pi],
                    gt_boxes=np.zeros((0, 4), dtype=np.float32),
                    gt_labels=None,
                    gt_indices=[],
                    match_info={
                        "matched": False,
                        "iou": 0.0,
                        "n_matched_total": n_matched,
                        "n_gt_total": n_gt,
                        "sum_iou_total": sum_iou_total,
                    },
                )
            segments.append(seg)

        return SlicingResult(
            segments=segments,
            all_pred_boxes=pred_boxes,
            all_pred_labels=pred_labels,
            all_pred_token_spans=token_spans,
            all_gt_boxes=gt_boxes,
            all_gt_labels=gt_labels,
        )
