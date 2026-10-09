from pathlib import Path
from typing import Any

from utils.io import prompt_text, read_json, write_json


def review_layers(
    original: Path, recomposed: Path, records: list[dict[str, Any]], api: Any, directory: Path, model: str
) -> dict[str, Any]:
    # 缺少必需图层时直接拒绝，不再调用模型。
    required = [r for r in records if r["role"] != "group_skip"]
    if not required or any(not r["generated"] for r in required):
        return {"acceptable": False, "reason": "Required layers are missing"}
    path = directory / "layer_review.json"
    if path.exists():
        return read_json(path)
    prompt = prompt_text(Path(__file__).parent / "prompts/review_layers.txt")
    result = api.json(prompt, [original, recomposed], model)
    if not isinstance(result.get("acceptable"), bool):
        raise ValueError("Layer review must return a boolean acceptance decision")
    write_json(path, result)
    return result
