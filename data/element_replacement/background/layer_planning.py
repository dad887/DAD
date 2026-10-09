from pathlib import Path
from typing import Any

from utils.io import prompt_text, read_json, slug, write_json


def plan_layers(image: Path, api: Any, directory: Path, model: str, max_layers: int) -> list[dict[str, Any]]:
    path = directory / "layer_plan.json"
    if path.exists():
        plan = read_json(path)
    else:
        prompt = prompt_text(Path(__file__).parent / "prompts/layer_plan.txt", max_layers=max_layers)
        plan = api.json(prompt, [image], model)
        write_json(path, plan)
    layers = plan["layers"]
    if not layers or len(layers) > max_layers:
        raise ValueError("Layer plan is empty or exceeds max_layers")
    result = []
    for i, layer in enumerate(layers):
        item = dict(layer)
        # 加序号前缀，保证图层编号唯一。
        item["layer_id"] = f"{i:02d}_{slug(item['layer_id'])}"
        item["z_order"] = int(item.get("z_order", i))
        if item["role"] not in {"base", "target", "group_skip"}:
            raise ValueError("Unknown background layer role")
        if item["element_type"] not in {"text", "visual"}:
            raise ValueError("Unknown background element type")
        result.append(item)
    if not any(layer["role"] == "base" for layer in result):
        raise ValueError("Layer plan must include a full-canvas base")
    return result
