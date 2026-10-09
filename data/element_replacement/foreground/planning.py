from pathlib import Path
from typing import Any

from utils.io import prompt_text, read_json, write_json


def normalize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    objects = []
    # 最多保留 4 个物体，第一个固定为主体；区域限制在画布内并避开顶部。
    for i, asset in enumerate(plan["selected_assets"][:4]):
        role = "focal" if i == 0 else asset["role"]
        if role not in {"focal", "activity", "supporting", "accent"}:
            raise ValueError("Unknown foreground role")
        scale = asset["scale"]
        if scale not in {"giant", "large", "medium", "small", "tiny"}:
            raise ValueError("Unknown foreground scale")
        label = asset["asset_label"].strip()
        if not label:
            raise ValueError("Foreground asset label is empty")
        if not label.lower().startswith("single "):
            label = "single " + label
        obj = {"role": role, "asset_label": label[:100], "scale": scale}
        if "region_norm" in asset:
            x, y, w, h = map(float, asset["region_norm"])
            w, h = min(max(w, 0.10), 0.62), min(max(h, 0.08), 0.58)
            x, y = min(max(x, 0.02), 0.98 - w), min(max(y, 0.18), 0.98 - h)
            obj["region_norm"] = [round(v, 4) for v in [x, y, w, h]]
        objects.append(obj)
    if len(objects) < 2:
        raise ValueError("Foreground planner must select at least two usable objects")
    return {
        "motif": plan["motif"],
        "style_directive": plan["style_directive"],
        "group_layout": plan["group_layout"],
        "candidate_assets": plan["candidate_assets"],
        "objects": objects,
    }


def plan_foreground(background: Path, spec: dict[str, Any], api: Any, directory: Path, model: str) -> dict[str, Any]:
    path = directory / "foreground_plan.json"
    if path.exists():
        return read_json(path)
    brief = "\n".join(
        f"{key}: {spec[key]}"
        for key in [
            "title_text",
            "theme_category",
            "theme",
            "style_prompt",
            "palette_hex",
            "layout_description",
            "safe_zones_norm",
        ]
    )
    prompt = prompt_text(Path(__file__).parent / "prompts/foreground_planner_prompt.txt", brief=brief)
    plan = normalize_plan(api.json(prompt, [background], model))
    write_json(path, plan)
    return plan
