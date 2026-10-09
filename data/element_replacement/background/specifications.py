from __future__ import annotations

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from background.prompt_builder import (
    constraints_for_mode,
    mode_sequence,
    prompt_from_spec,
)
from config import Config
from utils.io import error_text, prompt_text, read_json, write_json

LOGGER = logging.getLogger(__name__)

DESIGN_FIELDS = {
    "theme_category",
    "theme",
    "title_text",
    "title_position",
    "style_prompt",
    "palette_hex",
    "layout_description",
    "title_bbox_norm",
    "safe_zones_norm",
    "decor_layers",
}


def validate_designs(designs: Any) -> None:
    if not isinstance(designs, list) or len(designs) != 3:
        raise ValueError("Planner must return an array of exactly three designs")
    positions = {"top_compact", "bottom_compact", "side_rail", "corner_small", "center_small", "label_or_no_title"}
    for design in designs:
        if not isinstance(design, dict) or set(design) != DESIGN_FIELDS:
            raise ValueError("Each design must contain exactly the ten required fields")
        if design["title_position"] not in positions:
            raise ValueError("Unsupported title_position")
        for field in ["theme_category", "theme", "title_text", "style_prompt", "layout_description"]:
            if not isinstance(design[field], str) or not design[field].strip():
                raise ValueError(f"Missing design text: {field}")
        for field in ["palette_hex", "decor_layers", "safe_zones_norm"]:
            if not isinstance(design[field], list) or not design[field]:
                raise ValueError(f"Missing design list: {field}")
        for region in [design["title_bbox_norm"]] + design["safe_zones_norm"]:
            if not isinstance(region, list) or len(region) != 4:
                raise ValueError("Layout regions must use [x,y,w,h]")
            (x, y, w, h) = region
            if not (0 <= x < 1 and 0 <= y < 1 and (w > 0) and (h > 0) and (x + w <= 1.0001) and (y + h <= 1.0001)):
                raise ValueError("Layout region exceeds the canvas")


def assemble_spec(
    design: dict[str, Any], reference: dict[str, Any], mode: str, index: int, config: Config
) -> dict[str, Any]:
    spec = dict(design)
    spec.update(
        {
            "spec_id": f"background_{index:06d}",
            "generation_mode": mode,
            "image_model": config.image_model,
            "image_quality": config.image_quality,
            "image_size": config.canvas_size,
            "layer_constraints": constraints_for_mode(mode),
            "reference_id": reference["reference_id"],
            "reference_source_id": reference.get("source_id"),
            "reference_policy": "The reference image and annotations guide the LLM design planner. Background generation uses only its new design specification.",
        }
    )
    spec["image_prompt"] = prompt_from_spec(spec)
    return spec


def plan_specifications(
    references: list[dict[str, Any]], api: Any, config: Config
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    modes = mode_sequence(len(references) * 3, config.layer_friendly_ratio, random.Random(config.seed))
    template = prompt_text(Path(__file__).parent / "prompts/design_specifications.txt", canvas_size=config.canvas_size)

    def plan_one(item: tuple[int, dict[str, Any]]) -> list[dict[str, Any]]:
        reference_index, reference = item
        # 模式在规划前按比例分配；模型只返回设计字段，模式约束由程序追加。
        assigned = modes[reference_index * 3 : reference_index * 3 + 3]
        assignments = [
            {"specification_number": i + 1, "generation_mode": mode, "mode_requirements": constraints_for_mode(mode)}
            for i, mode in enumerate(assigned)
        ]
        prompt = (
            template
            + "\n\nPREASSIGNED MODES, IN OUTPUT ORDER:\n"
            + json.dumps(assignments)
            + "\nREFERENCE ANNOTATIONS:\n"
            + json.dumps(reference["annotations"], ensure_ascii=False)
        )
        path = config.output / "work/reference_plans" / f"{reference_index:06d}.json"
        if path.exists():
            designs = read_json(path)
        else:
            designs = api.json(prompt, [Path(reference["image"])], config.planner_model)
            validate_designs(designs)
            write_json(path, designs)
        return [
            assemble_spec(design, reference, assigned[i], reference_index * 3 + i, config)
            for i, design in enumerate(designs)
        ]

    # 单个参考规划失败只记录并跳过，不中止整批。
    specifications, failures = [], []
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = [(executor.submit(plan_one, item), item[1]) for item in enumerate(references)]
        for future, reference in futures:
            try:
                specifications.extend(future.result())
            except Exception as exc:
                failures.append({"reference_id": reference["reference_id"], "error": error_text(exc, api)})
                LOGGER.exception("Reference planning failed: %s", reference["reference_id"])
    write_json(config.output / "specifications.json", specifications)
    write_json(config.output / "reference_failures.json", failures)
    return specifications, failures
