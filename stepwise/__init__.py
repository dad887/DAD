"""
EleRPO: Element Relative Policy Optimization.

This package implements EleRPO, which combines sequence-level GRPO
advantages with fine-grained per-token advantages derived from the
incremental F1 gains of individual predicted elements.

Key components:
  - HungarianSlicing: Parses model output and matches predictions to GT.
  - SegmentIoUF1Reward: Computes per-segment IoU rewards.
  - compute_batch_segment_advantages: Computes prefix-conditioned delta-F1
    advantages with silhouette clustering.
  - build_per_token_advantages: Maps segment advantages to per-token tensors.
"""

from .segment import Segment, SlicingResult
from .slicing_strategy import HungarianSlicing
from .segment_reward import (
    SegmentIoUF1Reward,
    compute_batch_segment_advantages,
    build_per_token_advantages,
)

__all__ = [
    "Segment",
    "SlicingResult",
    "HungarianSlicing",
    "SegmentIoUF1Reward",
    "compute_batch_segment_advantages",
    "build_per_token_advantages",
]
