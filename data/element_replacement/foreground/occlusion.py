from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from foreground.layers import (
    make_object_layers,
    make_single_object_layer,
    transformed_soft_layer,
)
from foreground.metrics import (
    analyze_stack_fast,
    choose_soft_insert_cuts,
    foreground_head_intrusion_ratio,
    title_overlap_ratio,
)
from foreground.quality import candidate_score, quality_gate


def center_box(box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    (x0, y0, x1, y1) = box
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1), float(x1 - x0), float(y1 - y0))


def transform_boxes(
    boxes: list[tuple[int, int, int, int]],
    size: tuple[int, int],
    pull: float,
    scale: float,
    jitter: tuple[float, float],
) -> list[tuple[int, int, int, int]]:
    (W, H) = size
    centers = [center_box(b) for b in boxes]
    (fx, fy, _, _) = centers[0]
    target_x = fx + jitter[0] * W
    target_y = fy + jitter[1] * H
    out: list[tuple[int, int, int, int]] = []
    for i, (cx, cy, bw, bh) in enumerate(centers):
        role_scale = 1.0 if i == 0 else 1.03 if i == 1 else 1.08 if i == 2 else 1.02
        ncx = cx * (1.0 - pull) + target_x * pull
        ncy = cy * (1.0 - pull) + target_y * pull
        nw = bw * scale * role_scale
        nh = bh * scale * role_scale
        x0 = int(round(ncx - nw * 0.5))
        y0 = int(round(ncy - nh * 0.5))
        x1 = int(round(ncx + nw * 0.5))
        y1 = int(round(ncy + nh * 0.5))
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0
        if x1 > W:
            x0 -= x1 - W
            x1 = W
        if y1 > H:
            y0 -= y1 - H
            y1 = H
        out.append((max(0, x0), max(0, y0), min(W, x1), min(H, y1)))
    return out


def overlap_edge_boxes(
    boxes: list[tuple[int, int, int, int]], plan: dict[str, Any], size: tuple[int, int], variant: int
) -> list[tuple[int, int, int, int]]:
    (W, H) = size
    centers = [center_box(b) for b in boxes]
    out = list(boxes)

    def place(idx: int, anchor_idx: int, ox: float, oy: float, scale: float = 1.0) -> None:
        (acx, acy, aw, ah) = centers[anchor_idx]
        (_, _, bw, bh) = centers[idx]
        (nw, nh) = (bw * scale, bh * scale)
        ncx = acx + ox * (aw + nw) * 0.5
        ncy = acy + oy * (ah + nh) * 0.5
        x0 = int(round(ncx - nw * 0.5))
        y0 = int(round(ncy - nh * 0.5))
        x1 = int(round(ncx + nw * 0.5))
        y1 = int(round(ncy + nh * 0.5))
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0
        if x1 > W:
            x0 -= x1 - W
            x1 = W
        if y1 > H:
            y0 -= y1 - H
            y1 = H
        out[idx] = (max(0, x0), max(0, y0), min(W, x1), min(H, y1))

    roles = [str(obj.get("role")) for obj in plan["objects"]]
    activity_idx = next((i for (i, r) in enumerate(roles) if r == "activity"), None)
    supporting_idx = next((i for (i, r) in enumerate(roles) if r == "supporting"), None)
    accent_idx = next((i for (i, r) in enumerate(roles) if r == "accent"), None)
    presets = [
        {"activity": (-0.72, -0.04, 1.0), "supporting": (0.7, 0.54, 1.08), "accent": (-0.62, 0.38, 1.02)},
        {"activity": (-0.64, 0.1, 1.04), "supporting": (0.62, 0.32, 1.1), "accent": (0.58, 0.46, 1.04)},
        {"activity": (-0.52, -0.18, 1.0), "supporting": (0.48, 0.6, 1.14), "accent": (-0.55, 0.18, 1.06)},
        {"activity": (-0.82, 0.16, 1.03), "supporting": (0.78, 0.2, 1.12), "accent": (0.62, 0.36, 1.04)},
        {"activity": (-0.44, 0.02, 1.03), "supporting": (0.38, 0.48, 1.16), "accent": (-0.48, 0.48, 1.08)},
        {"activity": (-0.58, 0.28, 1.05), "supporting": (0.46, 0.18, 1.16), "accent": (0.44, 0.52, 1.08)},
    ]
    p = presets[variant % len(presets)]
    if activity_idx is not None:
        place(activity_idx, 0, *p["activity"])
    if supporting_idx is not None:
        place(supporting_idx, 0, *p["supporting"])
    if accent_idx is not None:
        place(accent_idx, activity_idx if activity_idx is not None and variant % 2 == 0 else 0, *p["accent"])
    return out


def optimize_occlusion(
    plan: dict[str, Any],
    base_layers: list[dict[str, Any]],
    soft_layers: list[dict[str, Any]],
    hard_layers: list[dict[str, Any]],
    maps: dict[str, Any],
    size: tuple[int, int],
    base_boxes: list[tuple[int, int, int, int]],
    max_candidates: int = 240,
) -> dict[str, Any]:
    # 枚举框位置、前后顺序、装饰插入位置和装饰移动，先过质量门槛，再按遮挡评分取最优。
    z_orders = [list(range(len(plan["objects"])))]
    z_orders.extend(
        [
            sorted(
                range(len(plan["objects"])),
                key=lambda i: {"focal": 0, "supporting": 1, "activity": 2, "accent": 3}.get(
                    plan["objects"][i].get("role"), i
                ),
            ),
            sorted(
                range(len(plan["objects"])),
                key=lambda i: {"activity": 0, "focal": 1, "supporting": 2, "accent": 3}.get(
                    plan["objects"][i].get("role"), i
                ),
            ),
            list(reversed(range(len(plan["objects"])))),
        ]
    )
    for perm in itertools.permutations(range(len(plan["objects"]))):
        if list(perm) not in z_orders:
            z_orders.append(list(perm))
    base_fg_by_idx = {
        i: make_single_object_layer(obj, i, box, size)
        for (i, (obj, box)) in enumerate(zip(plan["objects"], base_boxes))
    }
    candidate_box_sets: list[list[tuple[int, int, int, int]]] = []
    for variant in range(18):
        candidate_box_sets.append(overlap_edge_boxes(base_boxes, plan, size, variant))
        candidate_box_sets.append(
            overlap_edge_boxes(transform_boxes(base_boxes, size, 0.12, 1.06, (0.0, 0.02)), plan, size, variant)
        )
        candidate_box_sets.append(
            overlap_edge_boxes(transform_boxes(base_boxes, size, 0.22, 1.1, (0.02, 0.04)), plan, size, variant)
        )
    for pull in [0.0, 0.1, 0.18, 0.26, 0.34]:
        for scale in [1.0, 1.04, 1.08]:
            for jitter in [(0.0, 0.0), (-0.04, 0.0), (0.04, 0.0), (0.0, -0.04), (0.0, 0.04)]:
                candidate_box_sets.append(transform_boxes(base_boxes, size, pull, scale, jitter))
    evaluated: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, int, int, int], ...]] = set()
    cid = 0
    for boxes in candidate_box_sets:
        key = tuple(boxes)
        if key in seen:
            continue
        seen.add(key)
        if cid >= max_candidates:
            break
        for z_order in z_orders:
            fg_layers = make_object_layers(plan, boxes, size, z_order)
            if title_overlap_ratio(fg_layers, hard_layers) > 0.16:
                continue
            fg_union = np.zeros((size[1], size[0]), dtype=bool)
            for layer in fg_layers:
                fg_union |= layer["alpha"]
            for insert_cut in choose_soft_insert_cuts(soft_layers, fg_union, max_cuts=8):
                raw_above = soft_layers[insert_cut:]
                variant_ids = [0] if not raw_above else list(range(0, 9))
                for soft_variant in variant_ids:
                    if cid >= max_candidates:
                        break
                    above = [transformed_soft_layer(layer, fg_layers, size, soft_variant, 0.12) for layer in raw_above]
                    overlap_title = title_overlap_ratio(fg_layers, hard_layers)
                    head_intrusion = foreground_head_intrusion_ratio(fg_layers)
                    if overlap_title > 0.16 or head_intrusion > 0.6:
                        continue
                    # 层序：底图、插入点以下的装饰、前景、插入点以上的装饰、文字。
                    stack = base_layers + soft_layers[:insert_cut] + fg_layers + above + hard_layers
                    (all_stats, events) = analyze_stack_fast(
                        stack, size, target_groups={"fg", "soft", "hard"}, event_min_pair_ratio=0.1
                    )
                    fg_stats = [s for s in all_stats if s.get("layer_group") == "fg"]
                    (ok, qdiag) = quality_gate(base_fg_by_idx, fg_layers, above, fg_stats, all_stats, size)
                    if not ok:
                        continue
                    score = candidate_score(fg_stats, all_stats, overlap_title, len(events))
                    score -= 220.0 * (1.0 - float(qdiag.get("anchor_dice_min", 1.0)))
                    score -= 420.0 * float(qdiag.get("group_shift_ratio", 0.0))
                    score -= 260.0 * float(qdiag.get("moved_soft_area_ratio", 0.0))
                    evaluated.append(
                        {
                            "candidate_index": cid,
                            "score": score,
                            "boxes": boxes,
                            "z_order": z_order,
                            "soft_insert_cut": insert_cut,
                            "soft_variant": soft_variant,
                            "stats": fg_stats,
                            "all_target_stats": all_stats,
                            "occlusion_events": events,
                            "occlusion_event_count": len(events),
                            "quality_gate": qdiag,
                            "title_overlap_ratio": overlap_title,
                            "head_intrusion_ratio": head_intrusion,
                            "stack": stack,
                        }
                    )
                    # max_candidates 只统计通过质量门槛的候选。
                    cid += 1
                    # 只保留最优候选，避免累计全画布图层。
                    evaluated[:] = [max(evaluated, key=lambda candidate: candidate["score"])]
                if cid >= max_candidates:
                    break
            if cid >= max_candidates:
                break
    # 没有候选通过时使用初始布局，并在 quality_gate 中标记 fallback。
    if not evaluated:
        fg_layers = make_object_layers(plan, base_boxes, size, list(range(len(plan["objects"]))))
        stack = base_layers + fg_layers + soft_layers + hard_layers
        (all_stats, events) = analyze_stack_fast(
            stack, size, target_groups={"fg", "soft", "hard"}, event_min_pair_ratio=0.1
        )
        evaluated.append(
            {
                "candidate_index": 0,
                "score": -1000000000.0,
                "boxes": base_boxes,
                "z_order": list(range(len(plan["objects"]))),
                "soft_insert_cut": 0,
                "soft_variant": 0,
                "stats": [s for s in all_stats if s.get("layer_group") == "fg"],
                "all_target_stats": all_stats,
                "occlusion_events": events,
                "occlusion_event_count": len(events),
                "quality_gate": {"fallback": True},
                "stack": stack,
            }
        )
    evaluated.sort(key=lambda c: c["score"], reverse=True)
    return evaluated[0]
