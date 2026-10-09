from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from foreground.metrics import is_person
from utils.geometry import inner_canvas_mask, largest_connected_component
from utils.image_ops import fit_asset


def foreground_region_maps(
    base_layers: list[dict[str, Any]],
    soft_layers: list[dict[str, Any]],
    hard_layers: list[dict[str, Any]],
    size: tuple[int, int],
) -> dict[str, Any]:
    (W, H) = size
    hard = np.zeros((H, W), dtype=bool)
    soft = np.zeros((H, W), dtype=bool)
    for layer in hard_layers:
        hard |= layer["alpha"].astype(bool)
    for layer in soft_layers:
        soft |= layer["alpha"].astype(bool)
    valid = inner_canvas_mask(size, 0.03)
    free = valid & ~hard & ~soft
    (main, main_area, count) = largest_connected_component(free)
    return {"hard": hard, "soft": soft, "free": free, "main": main, "main_area": main_area, "component_count": count}


def target_box(size: str, W: int, H: int) -> tuple[int, int]:
    sizes = {
        "giant": (0.38, 0.36),
        "large": (0.32, 0.3),
        "medium": (0.24, 0.22),
        "small": (0.18, 0.16),
        "tiny": (0.12, 0.1),
    }
    (fw, fh) = sizes.get(size, sizes["medium"])
    return (max(80, round(W * fw)), max(80, round(H * fh)))


def paste_alpha_mask(layer: Image.Image, xy: tuple[int, int], size: tuple[int, int]) -> np.ndarray | None:
    (W, H) = size
    (x, y) = xy
    (x0, y0) = (max(0, x), max(0, y))
    (x1, y1) = (min(W, x + layer.width), min(H, y + layer.height))
    if x1 <= x0 or y1 <= y0:
        return None
    alpha = np.asarray(layer.getchannel("A"), dtype=np.float32) / 255.0
    crop = alpha[y0 - y : y1 - y, x0 - x : x1 - x]
    full = np.zeros((H, W), dtype=np.float32)
    full[y0:y1, x0:x1] = crop
    return full


def fit_group_regions_to_usable_space(
    objects: list[dict[str, Any]], maps: dict[str, Any], W: int, H: int, candidate_idx: int
) -> list[tuple[int, int, int, int]]:
    regions = [obj["region_norm"] for obj in objects]
    gx0 = min((r[0] for r in regions))
    gy0 = min((r[1] for r in regions))
    gx1 = max((r[0] + r[2] for r in regions))
    gy1 = max((r[1] + r[3] for r in regions))
    hard = maps["hard"].astype(np.float32)
    soft = maps["soft"].astype(np.float32)
    safe_mask = inner_canvas_mask((W, H), 0.03).astype(np.float32)
    free_main = maps["main"].astype(np.float32)
    (ys, xs) = np.where(maps["main"])
    if len(xs) > 0:
        main_cx = float(xs.mean())
        main_cy = float(ys.mean())
        (mx0, my0, mx1, my1) = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    else:
        main_cx = W * 0.5
        main_cy = H * 0.55
        (mx0, my0, mx1, my1) = (W * 0.03, H * 0.08, W * 0.97, H * 0.94)
    base_cx = (gx0 + gx1) / 2
    base_cy = (gy0 + gy1) / 2
    centers = [(x + w / 2, y + h / 2) for (x, y, w, h) in regions]
    # 保持规划的相对布局，搜索组中心、整体缩放、组内间距和点缀物偏移。
    role_scale = {"focal": 1.22, "activity": 1.4, "supporting": 1.14, "accent": 0.98}
    global_scales = [1.12, 1.22, 1.32, 1.42, 1.54]
    spread_scales = [1.08, 1.28, 1.48, 1.68, 1.88]
    shift_fracs = [
        (-0.12, -0.14),
        (-0.06, -0.1),
        (0.0, -0.1),
        (0.06, -0.06),
        (0.12, -0.02),
        (-0.04, 0.04),
        (0.04, 0.06),
    ]
    main_w = max(1.0, mx1 - mx0)
    main_h = max(1.0, my1 - my0)
    search_centers = [(main_cx + dx * main_w, main_cy + dy * main_h) for (dx, dy) in shift_fracs]
    search_centers.append((W * 0.5, H * 0.55))
    asset_cache: dict[str, Image.Image] = {}

    def candidate_boxes(
        group_cx: float, group_cy: float, global_scale: float, spread_scale: float, accent_shift: tuple[float, float]
    ) -> list[tuple[int, int, int, int]]:
        out = []
        for obj, (x, y, w, h), (cx, cy) in zip(objects, regions, centers):
            role = obj["role"]
            rscale = global_scale * role_scale.get(role, 1.0)
            nw = w * rscale
            nh = h * rscale
            ncx = group_cx / W + (cx - base_cx) * spread_scale
            ncy = group_cy / H + (cy - base_cy) * spread_scale
            if is_person(obj):
                ncx -= 0.05
                ncy -= 0.175
            elif role == "focal":
                ncx += 0.03
                ncy -= 0.025
            elif role == "supporting":
                ncx += 0.07
                ncy += 0.045
            elif role == "accent":
                ncx += 0.04 + accent_shift[0]
                ncy += -0.075 + accent_shift[1]
            out.append(
                (
                    round((ncx - nw / 2) * W),
                    round((ncy - nh / 2) * H),
                    round((ncx + nw / 2) * W),
                    round((ncy + nh / 2) * H),
                )
            )
        return out

    # 奖励落在主空白区的面积，重罚压住文字，并惩罚压住装饰和物体互相重叠。
    def box_score(boxes: list[tuple[int, int, int, int]]) -> float:
        score = 0.0
        alpha_items: list[tuple[dict[str, Any], np.ndarray, float]] = []
        for obj, (x0, y0, x1, y1) in zip(objects, boxes):
            if x0 < 0 or y0 < 0 or x1 > W or (y1 > H):
                return -1000000000000.0
            asset_path = obj.get("rgba_path")
            if not asset_path:
                return -1000000000000.0
            if asset_path not in asset_cache:
                asset_cache[asset_path] = Image.open(asset_path).convert("RGBA")
            (layer, xy) = fit_asset(asset_cache[asset_path], (x0, y0, x1, y1), 0.0)
            alpha = paste_alpha_mask(layer, xy, (W, H))
            if alpha is None:
                return -1000000000000.0
            mass = max(1.0, float(alpha.sum()))
            alpha_items.append((obj, alpha, mass))
            hard_overlap = float((alpha * hard).sum()) / mass
            soft_overlap = float((alpha * soft).sum()) / mass
            safe_overlap = float((alpha * safe_mask).sum()) / mass
            free_overlap = float((alpha * free_main).sum()) / mass
            role = str(obj.get("role"))
            area_reward = mass * (1.25 if role in {"focal", "activity"} else 0.95 if role == "supporting" else 0.34)
            score += area_reward * (0.45 + 0.65 * free_overlap + 0.2 * safe_overlap)
            score -= mass * (120.0 * hard_overlap + 1.6 * soft_overlap + 8.0 * (1.0 - safe_overlap))
        for i in range(len(alpha_items)):
            (obj_a, alpha_a, mass_a) = alpha_items[i]
            role_a = str(obj_a.get("role"))
            for j in range(i + 1, len(alpha_items)):
                (obj_b, alpha_b, mass_b) = alpha_items[j]
                role_b = str(obj_b.get("role"))
                overlap = float(np.minimum(alpha_a, alpha_b).sum())
                overlap_ratio = overlap / max(1.0, min(mass_a, mass_b))
                if overlap_ratio <= 0.015:
                    continue
                pair_weight = 22.0
                if "activity" in {role_a, role_b}:
                    pair_weight = 55.0
                if "accent" in {role_a, role_b}:
                    pair_weight *= 1.35
                score -= overlap * pair_weight
        return score

    scored: list[tuple[float, list[tuple[int, int, int, int]]]] = []
    seen: set[tuple[tuple[int, int, int, int], ...]] = set()
    accent_shifts = [(0.0, 0.0), (0.18, -0.18), (0.22, 0.06), (-0.18, -0.12), (-0.2, 0.1), (0.02, -0.28), (0.12, 0.22)]
    for group_cx, group_cy in search_centers:
        for global_scale in global_scales:
            for spread_scale in spread_scales:
                for accent_shift in accent_shifts:
                    boxes = candidate_boxes(group_cx, group_cy, global_scale, spread_scale, accent_shift)
                    key = tuple(boxes)
                    if key in seen:
                        continue
                    seen.add(key)
                    score = box_score(boxes)
                    if score > -100000000000.0:
                        scored.append((score, boxes))
    if not scored:
        return default_layout_boxes(objects, W, H)
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[min(candidate_idx, len(scored) - 1)][1]


def default_layout_boxes(objects: list[dict[str, Any]], W: int, H: int) -> list[tuple[int, int, int, int]]:
    centers = [(0.5, 0.57), (0.28, 0.62), (0.72, 0.68), (0.42, 0.42)]
    out = []
    for i, obj in enumerate(objects):
        (tw, th) = target_box(str(obj.get("scale", "medium")), W, H)
        (cx, cy) = centers[i % len(centers)]
        out.append((round(W * cx - tw / 2), round(H * cy - th / 2), round(W * cx + tw / 2), round(H * cy + th / 2)))
    return out


def layout_boxes(
    objects: list[dict[str, Any]], maps: dict[str, Any], size: tuple[int, int], candidate_idx: int
) -> list[tuple[int, int, int, int]]:
    (W, H) = size
    if all("region_norm" in obj for obj in objects):
        return fit_group_regions_to_usable_space(objects, maps, W, H, candidate_idx)
    return default_layout_boxes(objects, W, H)
