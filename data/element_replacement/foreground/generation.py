from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from utils.io import prompt_text
from utils.masking import alpha_bbox, remove_background_strict


def generate_assets(plan: dict[str, Any], api: Any, directory: Path, concurrency: int) -> dict[str, Any]:
    def generate(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, obj = item
        prompt = prompt_text(
            Path(__file__).parent / "prompts/object_image_prompt_template.txt",
            asset_label=obj["asset_label"],
            style_directive=plan["style_directive"],
        )
        raw_path = directory / "assets/raw" / f"{index:02d}.png"
        api.generate(prompt, raw_path, "1024x1024")
        rgba_path = directory / "assets/rgba" / raw_path.name
        rgba_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(raw_path) as raw:
            rgba = remove_background_strict(raw.convert("RGBA"), keep_multiple=False, pad=12)
        if alpha_bbox(rgba) is None:
            raise ValueError(f"Foreground asset is empty: {obj['asset_label']}")
        rgba.save(rgba_path)
        return {**obj, "raw_path": str(raw_path), "rgba_path": str(rgba_path), "image_prompt": prompt}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        objects = list(executor.map(generate, enumerate(plan["objects"])))
    return {**plan, "objects": objects}
