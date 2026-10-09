from __future__ import annotations

from typing import Any

import numpy as np

from utils.geometry import (
    bbox_area,
    bbox_intersection,
    bbox_to_int,
    content_bbox_from_mask,
    visible_bbox_from_crop,
)

PERSON_WORDS = (
    "person",
    "child",
    "student",
    "member",
    "maker",
    "worker",
    "reader",
    "teacher",
    "librarian",
    "volunteer",
    "adult",
)


def is_person(item: dict[str, Any]) -> bool:
    label = str(item.get("asset_label") or "").lower()
    return item.get("role") == "activity" or any(word in label for word in PERSON_WORDS)


def union_bbox_from_masks(masks: list[np.ndarray], fallback: tuple[int, int]) -> list[float]:
    (W, H) = fallback
    if not masks:
        return [0.0, 0.0, float(W), float(H)]
    union = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        union |= mask.astype(bool)
    box = content_bbox_from_mask(union)
    if box[2] <= box[0] or box[3] <= box[1]:
        return [0.0, 0.0, float(W), float(H)]
    return box


def target_id(item: dict[str, Any]) -> str:
    return str(item.get("object_id") or item.get("layer_id") or "unknown")


def is_stats_target(item: dict[str, Any], target_groups: set[str]) -> bool:
    if item.get("kind") == "fg":
        return "fg" in target_groups
    return str(item.get("layer_group") or "") in target_groups


def analyze_stack_fast(
    stack: list[dict[str, Any]], size: tuple[int, int], *, target_groups: set[str], event_min_pair_ratio: float = 0.1
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    (W, H) = size
    # 从最上层往下累积覆盖区域，得到每层的可见像素和可见框。
    covered = np.zeros((H, W), dtype=bool)
    stats_rev: list[dict[str, Any]] = []
    for item in reversed(stack):
        alpha = item["alpha"].astype(bool)
        content_bbox = item.get("content_bbox") or content_bbox_from_mask(alpha)
        item["content_bbox"] = content_bbox
        if is_stats_target(item, target_groups):
            (x0, y0, x1, y1) = bbox_to_int(content_bbox)
            (x0, y0) = (max(0, x0), max(0, y0))
            (x1, y1) = (min(W, x1), min(H, y1))
            if x1 <= x0 or y1 <= y0:
                mask_area = 0.0
                visible_area = 0.0
                visible_bbox = [content_bbox[0], content_bbox[1], content_bbox[0], content_bbox[1]]
            else:
                alpha_crop = alpha[y0:y1, x0:x1]
                visible_crop = alpha_crop & ~covered[y0:y1, x0:x1]
                mask_area = float(item.get("alpha_area", alpha_crop.sum()))
                visible_area = float(visible_crop.sum())
                visible_bbox = visible_bbox_from_crop(visible_crop, x0, y0, content_bbox)
            pixel_occ = 0.0 if mask_area <= 0 else max(0.0, 1.0 - visible_area / mask_area)
            ca = bbox_area(content_bbox)
            box_occ = 0.0 if ca <= 0 else max(0.0, 1.0 - bbox_area(visible_bbox) / ca)
            stats_rev.append(
                {
                    "target_id": target_id(item),
                    "object_id": item.get("object_id"),
                    "layer_id": item.get("layer_id"),
                    "target_kind": item.get("kind"),
                    "layer_group": item.get("layer_group"),
                    "role": item.get("role"),
                    "asset_label": item.get("asset_label") or item.get("layer_id"),
                    "bbox": content_bbox,
                    "visible_bbox": visible_bbox,
                    "mask_area": mask_area,
                    "canvas_area_ratio": mask_area / max(1.0, float(W * H)),
                    "visible_mask_area": visible_area,
                    "pixel_occlusion_ratio": pixel_occ,
                    "box_occlusion_ratio": box_occ,
                    "visible_ratio": 0.0 if mask_area <= 0 else visible_area / mask_area,
                }
            )
        if content_bbox[2] > content_bbox[0] and content_bbox[3] > content_bbox[1]:
            (x0, y0, x1, y1) = bbox_to_int(content_bbox)
            (x0, y0) = (max(0, x0), max(0, y0))
            (x1, y1) = (min(W, x1), min(H, y1))
            if x1 > x0 and y1 > y0:
                covered[y0:y1, x0:x1] |= alpha[y0:y1, x0:x1]
    # 记录每对图层之间超过阈值的遮挡关系。
    events: list[dict[str, Any]] = []
    target_indices = [i for (i, item) in enumerate(stack) if is_stats_target(item, target_groups)]
    for i in target_indices:
        target = stack[i]
        target_bbox = target.get("content_bbox") or content_bbox_from_mask(target["alpha"].astype(bool))
        area = max(1.0, float(target.get("alpha_area", target["alpha"].sum())))
        target_alpha = target["alpha"].astype(bool)
        for upper in stack[i + 1 :]:
            upper_bbox = upper.get("content_bbox") or content_bbox_from_mask(upper["alpha"].astype(bool))
            inter = bbox_intersection(target_bbox, upper_bbox)
            if inter is None:
                continue
            (x0, y0, x1, y1) = inter
            overlap = float((target_alpha[y0:y1, x0:x1] & upper["alpha"][y0:y1, x0:x1].astype(bool)).sum())
            ratio = overlap / area
            if ratio < event_min_pair_ratio:
                continue
            events.append(
                {
                    "target": target_id(target),
                    "target_kind": target.get("kind"),
                    "target_group": target.get("layer_group"),
                    "target_role": target.get("role"),
                    "occluder_kind": upper.get("kind"),
                    "occluder_group": upper.get("layer_group"),
                    "occluder_id": upper.get("object_id") or upper.get("layer_id"),
                    "pair_occlusion_ratio": ratio,
                    "overlap_pixels": overlap,
                }
            )
    return (list(reversed(stats_rev)), events)


def title_overlap_ratio(fg_layers: list[dict[str, Any]], hard_layers: list[dict[str, Any]]) -> float:
    if not fg_layers or not hard_layers:
        return 0.0
    hard = np.zeros_like(fg_layers[0]["alpha"], dtype=bool)
    for layer in hard_layers:
        hard |= layer["alpha"].astype(bool)
    fg = np.zeros_like(hard, dtype=bool)
    for layer in fg_layers:
        fg |= layer["alpha"].astype(bool)
    return float((fg & hard).sum()) / max(1.0, float(fg.sum()))


def foreground_head_intrusion_ratio(fg_layers: list[dict[str, Any]]) -> float:
    max_ratio = 0.0
    for i, layer in enumerate(fg_layers):
        if not is_person(layer):
            continue
        alpha = layer["alpha"].astype(bool)
        (x0, y0, x1, y1) = bbox_to_int(layer.get("content_bbox") or content_bbox_from_mask(alpha))
        if x1 <= x0 or y1 <= y0:
            continue
        (bw, bh) = (x1 - x0, y1 - y0)
        # 取人物框上部 35%、横向中间部分作为头部区域。
        hx0 = max(0, int(round(x0 + 0.22 * bw)))
        hx1 = min(alpha.shape[1], int(round(x1 - 0.22 * bw)))
        hy0 = max(0, y0)
        hy1 = min(alpha.shape[0], int(round(y0 + 0.35 * bh)))
        if hx1 <= hx0 or hy1 <= hy0:
            continue
        head = np.zeros_like(alpha, dtype=bool)
        head[hy0:hy1, hx0:hx1] = alpha[hy0:hy1, hx0:hx1]
        head_area = float(head.sum())
        if head_area <= 0:
            continue
        other = np.zeros_like(alpha, dtype=bool)
        for j, other_layer in enumerate(fg_layers):
            if i != j:
                other |= other_layer["alpha"].astype(bool)
        max_ratio = max(max_ratio, float((head & other).sum()) / head_area)
    return max_ratio


def choose_soft_insert_cuts(soft_layers: list[dict[str, Any]], fg_union: np.ndarray, max_cuts: int = 8) -> list[int]:
    ranked: list[tuple[float, int]] = []
    canvas_area = float(fg_union.size)
    for idx, layer in enumerate(soft_layers):
        alpha = layer["alpha"].astype(bool)
        area_ratio = float(alpha.sum()) / max(1.0, canvas_area)
        if area_ratio > 0.36:
            continue
        overlap = float((alpha & fg_union).sum()) / max(1.0, float(fg_union.sum()))
        if overlap <= 0.002:
            continue
        score = overlap * (1.0 - min(area_ratio, 0.28))
        ranked.append((score, idx))
    ranked.sort(reverse=True)
    cuts = {len(soft_layers)}
    for _, idx in ranked[: max(1, max_cuts - 1)]:
        cuts.add(idx)
    return sorted(cuts)


def mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a.astype(bool) & b.astype(bool)).sum())
    denom = float(a.sum() + b.sum())
    if denom <= 0:
        return 1.0
    return 2.0 * inter / denom
