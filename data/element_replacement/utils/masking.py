from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageFilter

ALPHA_THRESH = 10


def layer_text_for_rules(layer: dict[str, Any]) -> str:
    parts = [layer.get("layer_id"), layer.get("description_en"), layer.get("extraction_prompt_en")]
    return " ".join((str(p or "").lower() for p in parts))


def should_clear_hollow_frame_fill(layer: dict[str, Any], rgba: Image.Image) -> bool:
    if layer.get("role") == "base":
        return False
    include_text = layer_text_for_rules(layer)
    exclude_text = " ".join(
        (str(layer.get(key) or "").lower() for key in ("layer_id", "description_en", "description"))
    )
    include = ("border", "outline", "frame")
    exclude = (
        "label",
        "panel",
        "paper",
        "sticker",
        "text",
        "title",
        "plaque",
        "tab",
        "badge",
        "banner",
        "base",
        "background",
    )
    if not any((word in include_text for word in include)):
        return False
    if any((word in exclude_text for word in exclude)):
        return False
    alpha = np.array(rgba.getchannel("A"))
    coverage = float((alpha > 10).sum()) / float(alpha.size)
    return coverage > 0.4


def clear_internal_white_fill(rgba: Image.Image) -> Image.Image:
    arr = np.array(rgba.convert("RGBA")).copy()
    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3]
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    near_white = (maxc > 214) & (maxc - minc < 55) & (alpha > 10)
    arr[..., 3][near_white] = 0
    out = Image.fromarray(arr, "RGBA")
    out.putalpha(out.getchannel("A").filter(ImageFilter.MedianFilter(3)))
    return out


def remove_white_background(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    arr = np.array(img).copy()
    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3]
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    bg = (maxc > 214) & (maxc - minc < 55) | (alpha < 10)
    # 只清除与画布边缘连通的白底，保留物体内部的白色。
    flood = np.pad(bg.astype(np.uint8), 1, constant_values=1)
    cv2.floodFill(flood, None, (0, 0), 2, flags=4)
    visited = flood[1:-1, 1:-1] == 2
    arr[..., 3][visited] = 0
    out = Image.fromarray(arr, "RGBA")
    out.putalpha(out.getchannel("A").filter(ImageFilter.MedianFilter(3)))
    return out


def remove_matte_background(img: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    arr = np.array(img.convert("RGBA")).copy()
    distance = np.abs(arr[..., :3].astype(np.int16) - np.array(color, dtype=np.int16)).max(axis=2)
    background = (distance < 45) | (arr[..., 3] < 10)
    flood = np.pad(background.astype(np.uint8), 1, constant_values=1)
    cv2.floodFill(flood, None, (0, 0), 2, flags=4)
    arr[..., 3][flood[1:-1, 1:-1] == 2] = 0
    out = Image.fromarray(arr, "RGBA")
    out.putalpha(out.getchannel("A").filter(ImageFilter.MedianFilter(3)))
    return out


def connected_components(mask: np.ndarray) -> tuple[int, list[int]]:
    (num, labels, stats, _) = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    areas = sorted(
        [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num) if int(stats[i, cv2.CC_STAT_AREA]) > 100], reverse=True
    )
    large = [a for a in areas if a >= max(400, areas[0] * 0.18 if areas else 0)]
    return (len(large), areas)


def crop_to_alpha(img: Image.Image, pad: int = 10) -> Image.Image:
    alpha = np.asarray(img.getchannel("A"))
    (ys, xs) = np.where(alpha > 18)
    if len(xs) == 0:
        return img
    (x1, y1) = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad))
    (x2, y2) = (min(img.width, int(xs.max()) + 1 + pad), min(img.height, int(ys.max()) + 1 + pad))
    return img.crop((x1, y1, x2, y2))


def keep_largest_components(img: Image.Image, keep_ratio: float = 0.12) -> Image.Image:
    alpha = np.asarray(img.getchannel("A"))
    fg = alpha > 18
    (_, areas) = connected_components(fg)
    if not areas:
        return img
    threshold = max(180, int(areas[0] * keep_ratio))
    (num, labels, stats, _) = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=4)
    keep = np.zeros_like(fg, dtype=bool)
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) >= threshold:
            keep |= labels == i
    out = img.copy()
    out.putalpha(Image.fromarray(np.where(keep, alpha, 0).astype(np.uint8), "L"))
    return crop_to_alpha(out, 8)


def remove_background_strict(img: Image.Image, *, keep_multiple: bool, pad: int = 10) -> Image.Image:
    rgb = img.convert("RGB")
    arr = np.asarray(rgb).astype(np.int16)
    (h, w) = arr.shape[:2]
    k = max(8, min(h, w) // 16)
    edge = np.concatenate(
        [arr[:k].reshape(-1, 3), arr[-k:].reshape(-1, 3), arr[:, :k].reshape(-1, 3), arr[:, -k:].reshape(-1, 3)]
    )
    # 用四周边缘的中位色估计底色，接近底色或接近白色的像素置为透明。
    bg = np.median(edge, axis=0)
    diff = np.linalg.norm(arr - bg[None, None, :], axis=2)
    near_bg = diff < 38
    near_white = np.min(arr, axis=2) > 226
    alpha = np.clip((diff - 12) * 9, 0, 255).astype(np.uint8)
    alpha[near_bg | near_white] = 0
    alpha_img = Image.fromarray(alpha, "L").filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.GaussianBlur(0.6))
    out = img.convert("RGBA")
    out.putalpha(alpha_img)
    out = crop_to_alpha(out, pad)
    if not keep_multiple:
        out = keep_largest_components(out)
    return out


def alpha_bbox(img: Image.Image, threshold: int = 10) -> tuple[int, int, int, int] | None:
    arr = np.array(img.convert("RGBA"))
    (ys, xs) = np.where(arr[..., 3] > threshold)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def trim_alpha(img: Image.Image, pad: int = 4) -> Image.Image:
    img = img.convert("RGBA")
    bbox = alpha_bbox(img)
    if bbox is None:
        return img
    (x0, y0, x1, y1) = bbox
    return img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad)))
