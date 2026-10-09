import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from utils.geometry import content_bbox_from_mask
from utils.io import prompt_text, read_json, write_json
from utils.masking import ALPHA_THRESH

GROUP_ORDER = {"base": 0, "soft": 1, "hard": 2}


def layer_group(record: dict[str, Any]) -> str:
    # 文字和标签层受保护，始终在最上层；其余装饰可以被前景遮挡或移动。
    if record["role"] == "base":
        return "base"
    if record["element_type"] == "text" or re.search(r"\b(title|text|label)\b", record["description_en"].lower()):
        return "hard"
    return "soft"


def clean_background(
    records: list[dict[str, Any]], directory: Path, size: tuple[int, int]
) -> tuple[Path, list[dict[str, Any]]]:
    # 按后续合成的层序叠放（底图、装饰、文字），严格筛选看到的就是最终层序。
    generated = [r for r in records if r["generated"]]
    canvas = Image.new("RGBA", size, (255, 255, 255, 255))
    items = []
    for record in sorted(generated, key=lambda r: (GROUP_ORDER[layer_group(r)], r["z_order"])):
        with Image.open(record["rgba_path"]) as raw:
            rgba = raw.convert("RGBA")
        canvas.alpha_composite(rgba)
        alpha = np.asarray(rgba.getchannel("A")) > ALPHA_THRESH
        items.append(
            {
                "kind": "background",
                "layer_id": record["layer_id"],
                "record": record,
                "image": rgba,
                "alpha": alpha,
                "alpha_area": int(alpha.sum()),
                "content_bbox": content_bbox_from_mask(alpha),
            }
        )
    path = directory / "clean_background.png"
    canvas.convert("RGB").save(path)
    return path, items


def classify_layers(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups = {"base": [], "soft": [], "hard": []}
    for item in items:
        group = layer_group(item["record"])
        groups[group].append({**item, "layer_group": group})
    return groups["base"], groups["soft"], groups["hard"]


def screen_background(image: Path, spec: dict[str, Any], api: Any, directory: Path, model: str) -> dict[str, Any]:
    path = directory / "background_review.json"
    if path.exists():
        return read_json(path)
    brief = "\n".join(
        f"{key}: {spec[key]}"
        for key in [
            "theme_category",
            "theme",
            "title_text",
            "style_prompt",
            "palette_hex",
            "layout_description",
            "safe_zones_norm",
        ]
    )
    prompt = prompt_text(Path(__file__).parent / "prompts/screen_background.txt", brief=brief)
    result = api.json(prompt, [image], model)
    score = result.get("overall_score_1_to_5")
    if not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError("Background reviewer must return an integer score from 1 to 5")
    write_json(path, result)
    return result
