from __future__ import annotations

import math
from typing import Any

import numpy as np

from foreground.metrics import mask_dice
from utils.geometry import bbox_area, content_bbox_from_mask


def quality_gate(
    base_fg_by_idx: dict[int, dict[str, Any]],
    fg_layers: list[dict[str, Any]],
    above_soft_layers: list[dict[str, Any]],
    fg_stats: list[dict[str, Any]],
    all_stats: list[dict[str, Any]],
    size: tuple[int, int],
) -> tuple[bool, dict[str, float | str]]:
    (W, H) = size
    canvas_area = max(1.0, float(W * H))
    base_by_id = {layer["object_id"]: layer for layer in base_fg_by_idx.values()}
    dice_vals: list[float] = []
    weight_changes: list[float] = []
    for layer in fg_layers:
        base = base_by_id.get(layer["object_id"])
        if base is None:
            continue
        dice_vals.append(mask_dice(base["alpha"], layer["alpha"]))
        base_area = max(1.0, float(base["alpha"].sum()))
        weight_changes.append(abs(float(layer["alpha"].sum()) / base_area - 1.0))
    anchor_dice = min(dice_vals) if dice_vals else 1.0
    avg_weight_change = float(sum(weight_changes) / max(1, len(weight_changes)))
    fg_union = np.zeros((H, W), dtype=bool)
    base_union = np.zeros((H, W), dtype=bool)
    for layer in fg_layers:
        fg_union |= layer["alpha"].astype(bool)
    for layer in base_fg_by_idx.values():
        base_union |= layer["alpha"].astype(bool)
    fg_box = content_bbox_from_mask(fg_union)
    base_box = content_bbox_from_mask(base_union)
    (fg_cx, fg_cy) = (0.5 * (fg_box[0] + fg_box[2]), 0.5 * (fg_box[1] + fg_box[3]))
    (base_cx, base_cy) = (0.5 * (base_box[0] + base_box[2]), 0.5 * (base_box[1] + base_box[3]))
    group_shift = math.hypot(fg_cx - base_cx, fg_cy - base_cy) / (math.hypot(W, H) + 1e-06)
    fg_area_ratio = float(fg_union.sum()) / canvas_area
    fg_box_area_ratio = bbox_area(fg_box) / canvas_area
    moved_soft_area = sum(
        (float(layer["alpha"].sum()) / canvas_area for layer in above_soft_layers if layer.get("moved_from_layer_id"))
    )
    focal_or_activity_bad = 0
    fg_too_hidden = 0
    for s in fg_stats:
        vis = float(s["visible_ratio"])
        occ = float(s["pixel_occlusion_ratio"])
        if vis < 0.32:
            fg_too_hidden += 1
        if s.get("role") in {"focal", "activity"} and (vis < 0.48 or occ > 0.52):
            focal_or_activity_bad += 1
    big_soft_over_hidden = 0
    for s in all_stats:
        if (
            s.get("layer_group") == "soft"
            and float(s.get("canvas_area_ratio", 0.0)) >= 0.1
            and (float(s["visible_ratio"]) < 0.38)
        ):
            big_soft_over_hidden += 1
    diagnostics: dict[str, float | str] = {
        "anchor_dice_min": anchor_dice,
        "avg_weight_change": avg_weight_change,
        "group_shift_ratio": group_shift,
        "fg_area_ratio": fg_area_ratio,
        "fg_box_area_ratio": fg_box_area_ratio,
        "moved_soft_area_ratio": moved_soft_area,
        "focal_or_activity_bad_count": float(focal_or_activity_bad),
        "fg_too_hidden_count": float(fg_too_hidden),
        "big_soft_over_hidden_count": float(big_soft_over_hidden),
    }
    # 依次检查前景是否偏离初始布局、面积是否过大、主体是否被遮挡过多。
    if group_shift > 0.28:
        diagnostics["reject_reason"] = "group_shift"
        return (False, diagnostics)
    if avg_weight_change > 1.35:
        diagnostics["reject_reason"] = "weight_change"
        return (False, diagnostics)
    if fg_area_ratio > 0.42 or fg_box_area_ratio > 0.62:
        diagnostics["reject_reason"] = "fg_area"
        return (False, diagnostics)
    if focal_or_activity_bad > 0 or fg_too_hidden > 0:
        diagnostics["reject_reason"] = "foreground_visibility"
        return (False, diagnostics)
    if moved_soft_area > 0.28:
        diagnostics["reject_reason"] = "moved_soft_area"
        return (False, diagnostics)
    if big_soft_over_hidden > 0:
        diagnostics["reject_reason"] = "big_soft_hidden"
        return (False, diagnostics)
    diagnostics["reject_reason"] = "pass"
    return (True, diagnostics)


def candidate_score(
    fg_stats: list[dict[str, Any]], all_stats: list[dict[str, Any]], title_overlap: float, event_count: int
) -> float:
    ratios = [float(s["pixel_occlusion_ratio"]) for s in fg_stats]
    visible = [float(s["visible_ratio"]) for s in fg_stats]
    mid = sum((1 for r in ratios if 0.2 <= r <= 0.6))
    high_ok = sum((1 for (r, v) in zip(ratios, visible) if r > 0.6 and v >= 0.22))
    too_hidden = sum((1 for v in visible if v < 0.22))
    too_clean = sum((1 for r in ratios if r < 0.08))
    dist = sum((min(abs(r - 0.2), abs(r - 0.6), 0.0 if 0.2 <= r <= 0.6 else abs(r - 0.4)) for r in ratios))
    # 奖励 20%-60% 的中等遮挡和更多遮挡关系，惩罚过度遮挡、几乎无遮挡和压住标题。
    score = 1200.0 * mid + 260.0 * high_ok - 900.0 * too_hidden - 120.0 * too_clean
    all_ratios = [float(s["pixel_occlusion_ratio"]) for s in all_stats]
    all_mid = sum((1 for r in all_ratios if 0.1 <= r <= 0.7))
    soft_mid = sum(
        (1 for s in all_stats if s.get("layer_group") == "soft" and 0.1 <= float(s["pixel_occlusion_ratio"]) <= 0.7)
    )
    all_ge04 = sum((1 for r in all_ratios if r >= 0.4))
    soft_ge04 = sum(
        (1 for s in all_stats if s.get("layer_group") == "soft" and float(s["pixel_occlusion_ratio"]) >= 0.4)
    )
    fg_ge04 = sum((1 for s in fg_stats if float(s["pixel_occlusion_ratio"]) >= 0.4))
    score += 90.0 * min(all_mid, 10) + 55.0 * min(soft_mid, 6)
    score += 900.0 * min(all_ge04, 5) + 180.0 * max(0, all_ge04 - 5)
    score += 140.0 * min(soft_ge04, 4) + 120.0 * min(fg_ge04, 3)
    score += 150.0 * min(event_count, 8) + 30.0 * max(0, event_count - 8)
    score -= 1600.0 * max(0.0, title_overlap - 0.12)
    score -= 50.0 * dist
    for s in fg_stats:
        if s["role"] in {"activity", "focal"} and float(s["visible_ratio"]) < 0.36:
            score -= 300.0
        if s["role"] == "activity" and float(s["pixel_occlusion_ratio"]) > 0.45:
            score -= 260.0
    for s in all_stats:
        if (
            s.get("layer_group") == "soft"
            and float(s["pixel_occlusion_ratio"]) > 0.75
            and (float(s.get("canvas_area_ratio", 0.0)) > 0.06)
        ):
            score -= 260.0
    return score
