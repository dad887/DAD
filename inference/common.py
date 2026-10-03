"""Data, geometry, and output helpers shared by the public tools."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")
TYPE_MAP = {"t": "text", "v": "visual", "text": "text", "visual": "visual"}


def load_config(path=CONFIG_PATH):
    with Path(path).open() as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def resized_dimensions(width, height, max_size=2048):
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if max(width, height) <= max_size:
        return width, height
    if width >= height:
        return max_size, max(1, int(height * max_size / width))
    return max(1, int(width * max_size / height)), max_size


def prepare_image(path, max_size=2048, jpeg_quality=95):
    with Image.open(path) as original:
        image = original.convert("RGB")
    size = resized_dimensions(*image.size, max_size)
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    # Match the image serialization used by the public experiment.
    if jpeg_quality is not None:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    return image


def normalized_boxes(row, max_size=2048):
    width, height = row["width"], row["height"]
    new_w, new_h = resized_dimensions(width, height, max_size)
    out = []
    for box in row["boxes"]:
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError(f"Invalid box in {row.get('id', 'record')}")
        if (new_w, new_h) != (width, height):
            box = [
                int(box[0] * new_w / width),
                int(box[1] * new_h / height),
                int(box[2] * new_w / width),
                int(box[3] * new_h / height),
            ]
        out.append(
            [
                int(box[0] * 1000 / new_w),
                int(box[1] * 1000 / new_h),
                int(box[2] * 1000 / new_w),
                int(box[3] * 1000 / new_h),
            ]
        )
    return np.asarray(out, dtype=np.float32).reshape(-1, 4)


def parse_completion(text):
    text = re.sub(r"<\|[^>]+\|>", "", text)
    boxes, types = [], []
    for segment in text.strip().split(";"):
        fields = [x.strip() for x in segment.strip().split(",")]
        if len(fields) < 5 or fields[4] not in TYPE_MAP:
            continue
        try:
            box = [float(x) for x in fields[:4]]
        except ValueError:
            continue
        if not all(math.isfinite(x) for x in box):
            continue
        boxes.append(box)
        types.append(TYPE_MAP[fields[4]])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(
        types, dtype=object
    )
