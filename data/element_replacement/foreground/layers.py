from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from foreground.metrics import union_bbox_from_masks
from utils.geometry import content_bbox_from_mask
from utils.image_ops import fit_asset
from utils.masking import ALPHA_THRESH


def make_single_object_layer(
    obj: dict[str, Any], obj_idx: int, box: tuple[int, int, int, int], size: tuple[int, int]
) -> dict[str, Any]:
    (W, H) = size
    asset = Image.open(obj["rgba_path"]).convert("RGBA")
    (layer, xy) = fit_asset(asset, box, 0.0)
    full = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    full.alpha_composite(layer, xy)
    alpha = np.asarray(full.getchannel("A")) > ALPHA_THRESH
    return {
        "kind": "fg",
        "layer_group": "fg",
        "object_id": f"obj_{obj_idx:02d}",
        "role": obj.get("role"),
        "asset_label": obj.get("asset_label"),
        "image": full,
        "alpha": alpha,
        "alpha_area": int(alpha.sum()),
        "content_bbox": content_bbox_from_mask(alpha),
    }


def make_object_layers(
    plan: dict[str, Any], boxes: list[tuple[int, int, int, int]], size: tuple[int, int], z_order: list[int]
) -> list[dict[str, Any]]:
    by_idx = {
        i: make_single_object_layer(obj, i, box, size) for (i, (obj, box)) in enumerate(zip(plan["objects"], boxes))
    }
    return [by_idx[i] for i in z_order]


def transformed_soft_layer(
    layer: dict[str, Any], fg_layers: list[dict[str, Any]], size: tuple[int, int], variant: int, max_move_frac: float
) -> dict[str, Any]:
    if variant == 0:
        return layer
    (W, H) = size
    alpha = layer["alpha"].astype(bool)
    lb = content_bbox_from_mask(alpha)
    if lb[2] <= lb[0] or lb[3] <= lb[1]:
        return layer
    fb = union_bbox_from_masks([fg["alpha"] for fg in fg_layers], size)
    (fx0, fy0, fx1, fy1) = fb
    (fcx, fcy) = (0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1))
    (fw, fh) = (max(1.0, fx1 - fx0), max(1.0, fy1 - fy0))
    crop_box = tuple((int(round(v)) for v in lb))
    crop = layer["image"].crop(crop_box).convert("RGBA")
    (cw, ch) = crop.size
    if cw <= 0 or ch <= 0:
        return layer
    # 把装饰层移到前景组周围的预设位置，面积越大的装饰缩放和移动幅度越小。
    placements = [
        (-0.36, 0.18, 1.02),
        (0.36, 0.18, 1.02),
        (0.0, 0.34, 1.04),
        (-0.22, -0.08, 0.96),
        (0.22, -0.08, 0.96),
        (0.0, 0.04, 1.08),
        (-0.4, 0.36, 1.1),
        (0.4, 0.36, 1.1),
    ]
    (ox, oy, scale) = placements[(variant - 1) % len(placements)]
    canvas_area = max(1.0, float(W * H))
    layer_area_ratio = float(alpha.sum()) / canvas_area
    if layer_area_ratio >= 0.08:
        scale = min(scale, 1.04)
    if layer_area_ratio >= 0.16:
        scale = min(scale, 1.0)
    target_w = max(12, int(round(cw * scale)))
    target_h = max(12, int(round(ch * scale)))
    moved = crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
    (lcx, lcy) = (0.5 * (lb[0] + lb[2]), 0.5 * (lb[1] + lb[3]))
    cx = fcx + ox * fw
    cy = fcy + oy * fh
    local_max_move = max(0.0, float(max_move_frac))
    if layer_area_ratio >= 0.08:
        local_max_move *= 0.6
    if layer_area_ratio >= 0.16:
        local_max_move *= 0.45
    max_dx = local_max_move * W
    max_dy = local_max_move * H
    cx = max(lcx - max_dx, min(lcx + max_dx, cx))
    cy = max(lcy - max_dy, min(lcy + max_dy, cy))
    x0 = int(round(cx - target_w * 0.5))
    y0 = int(round(cy - target_h * 0.5))
    x0 = max(-target_w // 2, min(W - target_w // 2, x0))
    y0 = max(-target_h // 2, min(H - target_h // 2, y0))
    full = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    full.alpha_composite(moved, (x0, y0))
    out = dict(layer)
    out["layer_id"] = f"{layer['layer_id']}__moved{variant}"
    out["image"] = full
    out["alpha"] = np.asarray(full.getchannel("A")) > ALPHA_THRESH
    out["alpha_area"] = int(out["alpha"].sum())
    out["content_bbox"] = content_bbox_from_mask(out["alpha"])
    out["moved_from_layer_id"] = layer["layer_id"]
    return out
