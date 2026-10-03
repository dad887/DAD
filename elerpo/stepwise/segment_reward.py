"""
Per-segment reward computation and advantage estimation for EleRPO.

Core algorithm: **Delta-F1 prefix-conditioned advantage with silhouette
clustering** (``delta_f1_cond_prefix``).

For each rollout, boxes are considered in generation order.  The reward for
box *k* is the incremental F1 gain:

    ΔF1_k = F1(pred_{1..k}, GT) − F1(pred_{1..k−1}, GT)

To reduce variance, rollouts at position *k* are clustered by their prefix
F1 value ``F1(pred_{1..k−1})``, and the cluster-conditional mean ΔF1 is
subtracted.  Silhouette-based adaptive clustering automatically selects *k*
(number of clusters) or falls back to global mean when the data has no clear
cluster structure.

The resulting per-box advantages are mapped to per-token advantages and
combined with sequence-level GRPO advantages in a hybrid scheme.
"""

from typing import List, Optional, Tuple

import numpy as np

from .segment import Segment, SlicingResult
from .iou_utils import compute_hungarian_iou_f1_reward


# ---------------------------------------------------------------------------
# Per-segment reward
# ---------------------------------------------------------------------------

class SegmentIoUF1Reward:
    """
    IoU-based reward for individual segments (one predicted box each).

    For matched predictions the base reward is the IoU value (or 1.0 if
    ``use_iou_as_reward=False``).  Unmatched predictions receive 0.  An
    optional coverage penalty scales matched rewards by
    ``n_matched / n_gt``.

    Args:
        iou_threshold: Minimum IoU to consider a match valid.
        use_iou_as_reward: Use IoU value as base reward (vs 1.0).
        check_labels: Require label match for positive reward.
        coverage_penalty: Scale rewards by match coverage factor.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        use_iou_as_reward: bool = True,
        check_labels: bool = True,
        coverage_penalty: bool = True,
    ):
        self.iou_threshold = iou_threshold
        self.use_iou_as_reward = use_iou_as_reward
        self.check_labels = check_labels
        self.coverage_penalty = coverage_penalty

    def compute_rewards(self, slicing_result: SlicingResult) -> List[float]:
        """Return one reward value per segment."""
        return [self._compute_one(seg) for seg in slicing_result.segments]

    def _compute_one(self, segment: Segment) -> float:
        if not segment.has_predictions:
            return 0.0

        match_info = segment.match_info or {}
        is_matched = match_info.get("matched", False)
        iou = match_info.get("iou", 0.0)
        n_matched_total = match_info.get("n_matched_total", 0)
        n_gt_total = match_info.get("n_gt_total", 0)

        coverage_factor = n_matched_total / n_gt_total if n_gt_total > 0 else 0.0

        if not is_matched:
            return 0.0

        base_reward = iou if self.use_iou_as_reward else 1.0

        if self.check_labels and segment.pred_labels and segment.gt_labels:
            if segment.pred_labels[0] != segment.gt_labels[0]:
                base_reward = 0.0

        return base_reward * coverage_factor if self.coverage_penalty else base_reward


# ---------------------------------------------------------------------------
# Batch advantage computation
# ---------------------------------------------------------------------------

def compute_batch_segment_advantages(
    all_slicing_results: List[SlicingResult],
    all_segment_rewards: List[List[float]],
    num_generations: int = 1,
    norm_mode: str = "delta_f1_cond_prefix",
    im_end_recall: bool = False,
    reward_weights: Optional[List[float]] = None,
    n_clusters: int = 2,
    min_silhouette: float = 0.5,
) -> Tuple[List[List[float]], Optional[List[float]], Optional[dict]]:
    """
    Compute per-segment advantages for a batch of rollouts.

    Args:
        all_slicing_results: One SlicingResult per rollout.
        all_segment_rewards: One reward list per rollout.
        num_generations: Number of rollouts per prompt (K).
        norm_mode: Advantage normalization strategy.  Supported:
            ``"batch"`` — global z-score (baseline).
            ``"delta_f1"`` — incremental F1 with per-position z-score.
            ``"delta_f1_cond_prefix"`` — prefix-conditioned delta F1
            with silhouette clustering (recommended).
        im_end_recall: Assign recall-based advantage to the im_end token.
        reward_weights: Per-type weights for delta-F1 (e.g. [0.5, 0.5]).
        n_clusters: Max clusters for silhouette adaptive clustering.
        min_silhouette: Minimum silhouette score to use clustering.

    Returns:
        all_segment_advantages: Per-segment advantage lists.
        im_end_advantages: Per-sample im_end advantages (or None).
        cluster_stats: Clustering diagnostics (or None).
    """
    recall_rewards = None
    if im_end_recall:
        recall_rewards = []
        for sr in all_slicing_results:
            segs = [s for s in sr.segments if s.has_predictions]
            if segs and segs[0].match_info:
                n_m = segs[0].match_info.get("n_matched_total", 0)
                n_g = segs[0].match_info.get("n_gt_total", 0)
                recall_rewards.append(n_m / n_g if n_g > 0 else 0.0)
            else:
                recall_rewards.append(0.0)

    if norm_mode == "batch":
        advs, recall = _norm_batch(all_slicing_results, all_segment_rewards,
                                   recall_rewards, im_end_recall)
        return advs, recall, None
    elif norm_mode == "delta_f1":
        advs, im_end = _norm_delta_f1(
            all_slicing_results, all_segment_rewards, num_generations,
            reward_weights=reward_weights,
        )
        return advs, im_end, None
    elif norm_mode == "delta_f1_cond_prefix":
        advs, im_end, stats = _norm_delta_f1_cond_prefix(
            all_slicing_results, all_segment_rewards, num_generations,
            reward_weights=reward_weights,
            n_clusters=n_clusters,
            min_silhouette=min_silhouette,
        )
        return advs, im_end, stats
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")


# ---------------------------------------------------------------------------
# Per-token advantage broadcasting
# ---------------------------------------------------------------------------

def build_per_token_advantages(
    slicing_result: SlicingResult,
    segment_advantages: List[float],
    seq_len: int,
) -> np.ndarray:
    """
    Map segment advantages to a per-token advantage vector.

    Each token in a segment's token span receives that segment's advantage.
    Tokens outside any segment span get 0.

    Args:
        slicing_result: SlicingResult with segment token spans.
        segment_advantages: One advantage value per segment.
        seq_len: Total number of tokens in the sequence.

    Returns:
        (seq_len,) float32 array of per-token advantages.
    """
    per_token = np.zeros(seq_len, dtype=np.float32)

    for seg, adv in zip(slicing_result.segments, segment_advantages):
        if seg.segment_token_span is not None:
            start, end = seg.segment_token_span
            start = max(0, min(start, seq_len))
            end = max(0, min(end, seq_len))
            if start < end:
                per_token[start:end] = adv

    return per_token


# ---------------------------------------------------------------------------
# Normalization: batch baseline
# ---------------------------------------------------------------------------

def _norm_batch(all_slicing_results, all_segment_rewards, recall_rewards,
                im_end_recall):
    """Global z-score across all segments in the batch."""
    all_flat = []
    for sr, rewards in zip(all_slicing_results, all_segment_rewards):
        for seg, r in zip(sr.segments, rewards):
            if seg.has_predictions:
                all_flat.append(r)
    if im_end_recall and recall_rewards:
        all_flat.extend(recall_rewards)

    if not all_flat:
        return ([[0.0] * len(rw) for rw in all_segment_rewards],
                [0.0] * len(all_slicing_results) if im_end_recall else None)

    mean_r = np.mean(all_flat)
    std_r = np.std(all_flat) + 1e-8

    all_advs = []
    for sr, rewards in zip(all_slicing_results, all_segment_rewards):
        advs = []
        for seg, r in zip(sr.segments, rewards):
            advs.append((r - mean_r) / std_r if seg.has_predictions else 0.0)
        all_advs.append(advs)

    recall_advs = None
    if im_end_recall and recall_rewards:
        recall_advs = [(rr - mean_r) / std_r for rr in recall_rewards]

    return all_advs, recall_advs


# ---------------------------------------------------------------------------
# Normalization: delta-F1 (base)
# ---------------------------------------------------------------------------

def _compute_weighted_reward(pred_boxes_list, pred_labels_list,
                             gt_boxes, gt_labels, reward_weights):
    """Type-separated weighted Hungarian IoU F1."""
    if not pred_boxes_list:
        return 0.0

    pred_arr = np.array(pred_boxes_list, dtype=np.float32)

    has_labels = (
        pred_labels_list
        and any(l is not None for l in pred_labels_list)
        and gt_labels is not None
        and reward_weights is not None
        and len(reward_weights) >= 2
    )

    if not has_labels:
        return compute_hungarian_iou_f1_reward(gt_boxes, pred_arr, min_iou=0.0, beta=1.0)

    gt_labels_arr = np.array(gt_labels)
    pred_labels_arr = np.array(pred_labels_list)

    gt_vis = gt_boxes[gt_labels_arr == "visual"] if (gt_labels_arr == "visual").any() else np.zeros((0, 4), dtype=np.float32)
    gt_txt = gt_boxes[gt_labels_arr == "text"] if (gt_labels_arr == "text").any() else np.zeros((0, 4), dtype=np.float32)
    pred_vis = pred_arr[pred_labels_arr == "visual"] if (pred_labels_arr == "visual").any() else np.zeros((0, 4), dtype=np.float32)
    pred_txt = pred_arr[pred_labels_arr == "text"] if (pred_labels_arr == "text").any() else np.zeros((0, 4), dtype=np.float32)

    f1_vis = compute_hungarian_iou_f1_reward(gt_vis, pred_vis, min_iou=0.0, beta=1.0)
    f1_txt = compute_hungarian_iou_f1_reward(gt_txt, pred_txt, min_iou=0.0, beta=1.0)
    return reward_weights[0] * f1_vis + reward_weights[1] * f1_txt


def _build_prefix_entries(all_slicing_results, reward_weights):
    """Shared helper: build per-rollout prefix-F1 and delta-F1 entries."""
    n_samples = len(all_slicing_results)
    all_prefix_f1 = []
    all_entries = []

    for i in range(n_samples):
        sr = all_slicing_results[i]
        gt_boxes = sr.all_gt_boxes
        gt_labels = sr.all_gt_labels
        if gt_boxes is None:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)

        segs_with_preds = [(si, seg) for si, seg in enumerate(sr.segments)
                           if seg.has_predictions]

        prefix_f1 = [0.0]
        entries = []
        prev_reward = 0.0
        pred_boxes_so_far = []
        pred_labels_so_far = []

        for k, (seg_idx, seg) in enumerate(segs_with_preds):
            if seg.n_pred > 0:
                pred_boxes_so_far.append(seg.pred_boxes[0])
                label = seg.pred_labels[0] if seg.pred_labels else None
                pred_labels_so_far.append(label)

            reward = _compute_weighted_reward(
                pred_boxes_so_far, pred_labels_so_far,
                gt_boxes, gt_labels, reward_weights,
            )
            prefix_f1.append(reward)
            entries.append(("box", seg_idx, k, reward - prev_reward))
            prev_reward = reward

        im_end_k = len(segs_with_preds)
        entries.append(("im_end", -1, im_end_k, 0.0))

        all_prefix_f1.append(prefix_f1)
        all_entries.append(entries)

    return all_prefix_f1, all_entries


def _norm_delta_f1(all_slicing_results, all_segment_rewards, num_generations,
                   reward_weights=None):
    """
    Delta-F1 advantage: per-position z-score of incremental F1 gain.

    At each position *k*, ΔF1 values from all rollouts in the same
    generation group are z-scored together.
    """
    n_samples = len(all_slicing_results)
    _, all_entries = _build_prefix_entries(all_slicing_results, reward_weights)

    all_advs = [[0.0] * len(rw) for rw in all_segment_rewards]
    im_end_advs = [0.0] * n_samples

    for group_start in range(0, n_samples, num_generations):
        group_end = min(group_start + num_generations, n_samples)

        pos_to_entries: dict = {}
        for i in range(group_start, group_end):
            for etype, seg_idx, k, dr in all_entries[i]:
                pos_to_entries.setdefault(k, []).append((i, etype, seg_idx, dr))

        for k, entries in pos_to_entries.items():
            if len(entries) < 2:
                continue
            values = [dr for _, _, _, dr in entries]
            mean_v = np.mean(values)
            std_v = np.std(values) + 1e-8
            for sample_idx, etype, seg_idx, dr in entries:
                adv = (dr - mean_v) / std_v
                if etype == "box":
                    all_advs[sample_idx][seg_idx] = adv
                else:
                    im_end_advs[sample_idx] = adv

    return all_advs, im_end_advs


# ---------------------------------------------------------------------------
# Normalization: delta-F1 prefix-conditioned (main contribution)
# ---------------------------------------------------------------------------

def _kmeans_cluster(values, n_clusters, max_iter=30, allow_singletons=False):
    """1-D k-means clustering on scalar values.

    Returns an int array of cluster assignments, or None if any cluster
    would have fewer than 2 members (unless ``allow_singletons=True``).
    """
    n = len(values)
    if n < 2 * n_clusters:
        return None

    vals = np.asarray(values, dtype=np.float64)
    quantiles = [(2 * c + 1) / (2 * n_clusters) for c in range(n_clusters)]
    centroids = np.quantile(vals, quantiles)

    assignments = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = np.abs(vals[:, None] - centroids[None, :])
        new_assignments = np.argmin(dists, axis=1)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for c in range(n_clusters):
            mask = assignments == c
            if mask.any():
                centroids[c] = vals[mask].mean()

    if not allow_singletons:
        for c in range(n_clusters):
            if np.sum(assignments == c) < 2:
                return None

    return assignments


def _silhouette_score_1d(vals, assignments, n_clusters):
    """Mean silhouette score for 1-D clustered data."""
    n = len(vals)
    scores = np.zeros(n)
    indices = np.arange(n)
    for i in range(n):
        same = assignments == assignments[i]
        n_same = same.sum()
        if n_same <= 1:
            scores[i] = 0.0
            continue
        a_i = np.mean(np.abs(vals[same & (indices != i)] - vals[i]))

        b_i = np.inf
        for c in range(n_clusters):
            if c == assignments[i]:
                continue
            other = assignments == c
            if other.any():
                b_i = min(b_i, np.mean(np.abs(vals[other] - vals[i])))

        denom = max(a_i, b_i, 1e-8)
        scores[i] = (b_i - a_i) / denom
    return float(np.mean(scores))


def _silhouette_adaptive_cluster(values, max_k=4, min_score=0.5):
    """Adaptive 1-D clustering using silhouette score to choose *k*.

    Tries k = 2 … max_k via k-means and picks the *k* with the highest
    mean silhouette score.  Returns ``(None, info)`` when the best score
    is below ``min_score``, meaning the data has no clear cluster structure
    and a global baseline should be used instead.

    Returns:
        (assignments_or_None, info_dict)
    """
    n = len(values)
    if n < 4:
        return None, {"chosen_k": 0, "best_score": 0.0}

    vals = np.asarray(values, dtype=np.float64)

    best_assignments = None
    best_score = min_score
    best_k = 0
    for k in range(2, min(max_k + 1, n // 2 + 1)):
        assignments = _kmeans_cluster(vals, k, allow_singletons=True)
        if assignments is None:
            continue
        if len(set(assignments)) < 2:
            continue
        score = _silhouette_score_1d(vals, assignments, k)
        if score > best_score:
            best_score = score
            best_assignments = assignments
            best_k = k

    return best_assignments, {"chosen_k": best_k, "best_score": float(best_score)}


def _norm_delta_f1_cond_prefix(
    all_slicing_results, all_segment_rewards, num_generations,
    reward_weights=None,
    n_clusters=2,
    min_silhouette=0.5,
):
    """
    Prefix-conditioned delta-F1 advantage with silhouette clustering.

    At each position *k*, rollouts are clustered by their prefix F1 value
    ``F1(pred_{1..k−1})``.  The advantage is the residual after subtracting
    the cluster-conditional mean ΔF1:

        r̂_k^(i) = ΔF1_k^(i) − mean_{j ∈ C(i)} ΔF1_k^(j)

    Silhouette-based adaptive clustering tries k = 2 … ``n_clusters`` and
    picks the k with the best silhouette score, falling back to global mean
    when the score is below ``min_silhouette``.

    Finally, scale normalization divides by per-position RMS.

    Args:
        all_slicing_results: One SlicingResult per rollout.
        all_segment_rewards: One reward list per rollout.
        num_generations: Rollouts per prompt (K).
        reward_weights: Type-separated F1 weights (e.g. [0.5, 0.5]).
        n_clusters: Maximum number of clusters to try.
        min_silhouette: Minimum silhouette score to apply clustering.

    Returns:
        (all_segment_advantages, im_end_advantages, cluster_stats)
    """
    n_samples = len(all_slicing_results)
    all_prefix_f1, all_entries = _build_prefix_entries(
        all_slicing_results, reward_weights
    )

    all_advs = [[0.0] * len(rw) for rw in all_segment_rewards]
    im_end_advs = [0.0] * n_samples

    cluster_chosen_ks = []
    cluster_scores = []
    cluster_used_count = 0
    cluster_total_count = 0

    for group_start in range(0, n_samples, num_generations):
        group_end = min(group_start + num_generations, n_samples)

        pos_to_entries: dict = {}
        for i in range(group_start, group_end):
            for etype, seg_idx, k, dr in all_entries[i]:
                pos_to_entries.setdefault(k, []).append((i, etype, seg_idx, dr))

        for k, entries in pos_to_entries.items():
            n_at_k = len(entries)
            if n_at_k < 2:
                continue

            rollout_ids = [e[0] for e in entries]
            delta_vals = np.array([e[3] for e in entries])

            pf1_before = np.array([
                all_prefix_f1[i][k]
                if k < len(all_prefix_f1[i])
                else all_prefix_f1[i][-1]
                for i in rollout_ids
            ])

            clusters, cluster_info = _silhouette_adaptive_cluster(
                pf1_before, max_k=n_clusters, min_score=min_silhouette
            )

            cluster_total_count += 1
            if clusters is not None:
                cluster_used_count += 1
                cluster_chosen_ks.append(cluster_info["chosen_k"])
                cluster_scores.append(cluster_info["best_score"])
            else:
                cluster_chosen_ks.append(1)

            r_hat = np.zeros(n_at_k)
            for j in range(n_at_k):
                mask = (clusters == clusters[j]) if clusters is not None else np.ones(n_at_k, dtype=bool)
                r_hat[j] = delta_vals[j] - np.mean(delta_vals[mask])

            rms = np.sqrt(np.mean(r_hat ** 2)) + 1e-8
            normalized = r_hat / rms

            for j, (sample_idx, etype, seg_idx, _) in enumerate(entries):
                if etype == "box":
                    all_advs[sample_idx][seg_idx] = float(normalized[j])
                else:
                    im_end_advs[sample_idx] = float(normalized[j])

    cluster_stats = {
        "cluster_used_frac": cluster_used_count / max(cluster_total_count, 1),
        "cluster_k_mean": float(np.mean(cluster_chosen_ks)) if cluster_chosen_ks else 0.0,
        "cluster_silhouette_mean": float(np.mean(cluster_scores)) if cluster_scores else 0.0,
        "cluster_total_positions": cluster_total_count,
    }

    return all_advs, im_end_advs, cluster_stats
