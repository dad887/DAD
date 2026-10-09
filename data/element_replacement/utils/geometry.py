from __future__ import annotations

import math

import cv2
import numpy as np


def content_bbox_from_mask(mask: np.ndarray) -> list[float]:
    (ys, xs) = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def bbox_area(b: list[float] | tuple[float, ...] | None) -> float:
    if not b or len(b) < 4:
        return 0.0
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))


def bbox_to_int(b: list[float] | tuple[float, ...]) -> tuple[int, int, int, int]:
    return (int(math.floor(b[0])), int(math.floor(b[1])), int(math.ceil(b[2])), int(math.ceil(b[3])))


def bbox_intersection(a: list[float], b: list[float]) -> tuple[int, int, int, int] | None:
    (ax0, ay0, ax1, ay1) = bbox_to_int(a)
    (bx0, by0, bx1, by1) = bbox_to_int(b)
    (x0, y0) = (max(ax0, bx0), max(ay0, by0))
    (x1, y1) = (min(ax1, bx1), min(ay1, by1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def visible_bbox_from_crop(crop: np.ndarray, x0: int, y0: int, fallback: list[float]) -> list[float]:
    rows = crop.any(axis=1)
    if not rows.any():
        return [fallback[0], fallback[1], fallback[0], fallback[1]]
    cols = crop.any(axis=0)
    return [
        float(x0 + int(cols.argmax())),
        float(y0 + int(rows.argmax())),
        float(x0 + crop.shape[1] - int(cols[::-1].argmax())),
        float(y0 + crop.shape[0] - int(rows[::-1].argmax())),
    ]


def inner_canvas_mask(size: tuple[int, int], margin_ratio: float = 0.03) -> np.ndarray:
    (W, H) = size
    mask = np.zeros((H, W), dtype=bool)
    mx = round(W * margin_ratio)
    my = round(H * margin_ratio)
    mask[my : H - my, mx : W - mx] = True
    return mask


def largest_connected_component(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    (num_labels, labels, stats, _) = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, dtype=bool)
    if num_labels <= 1:
        return (out, 0, 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = int(np.argmax(areas)) + 1
    out = labels == best_idx
    return (out, int(areas.max()), int(num_labels - 1))
