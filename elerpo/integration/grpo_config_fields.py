"""
New GRPOConfig fields required for EleRPO.

Add these fields to your ms-swift ``GRPOConfig`` (``arguments.py``) dataclass.
They are read by the trainer during ``_init_stepwise_training()`` and the
hybrid advantage computation path.

Usage in ms-swift:
    In ``swift/trainers/arguments.py``, add these fields to the
    ``GRPOConfig`` dataclass alongside the existing GRPO fields.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class StepwiseGRPOFields:
    """
    Stepwise training fields to add to GRPOConfig.

    Copy these fields into your GRPOConfig dataclass.
    """

    # --- Core stepwise toggle ---

    stepwise_training: bool = False
    """Enable stepwise per-segment reward computation.  When True, the
    trainer slices each rollout into per-box segments and computes
    fine-grained per-token advantages."""

    slicing_strategy: Literal["hungarian"] = "hungarian"
    """How to partition predictions and GT into segments.
    ``hungarian`` uses optimal 1-to-1 Hungarian matching."""

    stepwise_response_format: str = "semicolon_with_type"
    """Format of the model's text output for parsing into boxes.
    Default ``semicolon_with_type`` expects ``x1,y1,x2,y2,t;...``."""

    # --- Per-segment reward parameters ---

    stepwise_iou_threshold: float = 0.5
    """Minimum IoU to consider a prediction–GT match valid."""

    stepwise_use_iou_reward: bool = True
    """Use IoU value as the base reward (vs 1.0 for any valid match)."""

    stepwise_check_labels: bool = True
    """Require label match (e.g. text vs visual) for positive reward."""

    stepwise_include_delimiter: bool = False
    """Extend each box's token span to include the trailing semicolon."""

    # --- Advantage normalization ---

    stepwise_norm_mode: Literal[
        "batch",
        "delta_f1",
        "delta_f1_cond_prefix",
    ] = "batch"
    """Advantage normalization strategy.

    - ``batch``: global z-score across all segments (baseline).
    - ``delta_f1``: incremental F1 with per-position z-score.
    - ``delta_f1_cond_prefix``: prefix-conditioned ΔF1 with silhouette
      clustering (recommended).
    """

    stepwise_delta_f1_weights: Optional[List[float]] = None
    """Type-separated reward weights for delta-F1 computation.
    E.g. ``[0.5, 0.5]`` for equal visual/text weighting."""

    stepwise_n_clusters: int = 2
    """Maximum number of clusters for silhouette adaptive clustering."""

    stepwise_min_silhouette: float = 0.5
    """Minimum silhouette score to apply clustering.  Below this threshold,
    the algorithm falls back to global mean (no clustering)."""

    # --- Hybrid combination ---

    stepwise_hybrid_mode: Literal["average", "element_only"] = "average"
    """How to combine sequence-level and element-level advantages.

    - ``average``: ``0.5 * seq_advantage + 0.5 * box_advantage``.
    - ``element_only``: use only element-level advantages.
    """

    stepwise_im_end_recall: bool = False
    """Assign recall-based advantage to the im_end token."""
