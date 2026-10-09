import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from utils.io import prompt_text, write_json
from utils.masking import (
    ALPHA_THRESH,
    alpha_bbox,
    clear_internal_white_fill,
    remove_matte_background,
    remove_white_background,
    should_clear_hollow_frame_fill,
)


def extract_layers(
    image: Path, layers: list[dict[str, Any]], api: Any, directory: Path, concurrency: int
) -> tuple[Path, list[dict[str, Any]]]:
    with Image.open(image) as source:
        size = source.size
    image_size = f"{size[0]}x{size[1]}"

    def extract(layer: dict[str, Any]) -> dict[str, Any]:
        if layer["role"] == "group_skip":
            return {**layer, "generated": False, "skip_reason": "group_skip"}
        # 白色元素使用灰底提取，避免与白色底图一起被清除。
        gray_matte = layer["role"] != "base" and bool(re.search(r"\bwhite\b", layer["description_en"], re.IGNORECASE))
        suffix = "_gray" if gray_matte else ""
        raw_path = directory / "layers/raw" / f"{layer['layer_id']}{suffix}.png"
        prompt = prompt_text(
            Path(__file__).parent / "prompts/extract_layer.txt", extraction_prompt=layer["extraction_prompt_en"]
        )
        if gray_matte:
            prompt = re.sub(
                r"\b(?:plain\s+)?white background\b",
                "plain solid neutral gray background (#808080)",
                prompt,
                flags=re.IGNORECASE,
            )
            prompt += "\nPreserve the requested white element as white. The gray matte is only for extraction and must be uniform #808080."
        api.extract(prompt, image, raw_path, image_size)
        with Image.open(raw_path) as raw:
            if raw.size != size:
                raise ValueError("Extracted layer size differs from background")
            rgba = raw.convert("RGBA")
        if layer["role"] != "base":
            rgba = remove_matte_background(rgba, (128, 128, 128)) if gray_matte else remove_white_background(rgba)
            # 边框类图层的内部常被填成白色，清空后才能露出下层内容。
            if should_clear_hollow_frame_fill(layer, rgba):
                rgba = clear_internal_white_fill(rgba)
        rgba_path = directory / "layers/rgba" / raw_path.name
        rgba_path.parent.mkdir(parents=True, exist_ok=True)
        rgba.save(rgba_path)
        bbox = alpha_bbox(rgba)
        if bbox is None:
            raise ValueError(f"Required layer is empty: {layer['layer_id']}")
        mask_path = directory / "layers/masks" / raw_path.name
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        rgba.getchannel("A").save(mask_path)
        return {
            **layer,
            "generated": True,
            "raw_path": str(raw_path),
            "rgba_path": str(rgba_path),
            "mask_path": str(mask_path),
            "bbox": bbox,
            "alpha_area": int((np.asarray(rgba.getchannel("A")) > ALPHA_THRESH).sum()),
            "matte_color": "#808080" if gray_matte else "#FFFFFF",
        }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        records = list(executor.map(extract, layers))
    canvas = Image.new("RGBA", size, (255, 255, 255, 255))
    for layer in sorted(records, key=lambda x: x["z_order"]):
        if layer["generated"]:
            with Image.open(layer["rgba_path"]) as rgba:
                canvas.alpha_composite(rgba.convert("RGBA"))
    recomposed = directory / "recomposed.png"
    canvas.convert("RGB").save(recomposed)
    write_json(directory / "layer_records.json", records)
    return recomposed, records
