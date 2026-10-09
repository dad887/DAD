from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from api import ModelClient
from background.generation import generate_background
from background.layer_extraction import extract_layers
from background.layer_planning import plan_layers
from background.layer_review import review_layers
from background.screening import classify_layers, clean_background, screen_background
from background.specifications import plan_specifications
from config import Config, parse_args
from foreground.export import save_annotation
from foreground.generation import generate_assets
from foreground.layout import foreground_region_maps, layout_boxes
from foreground.occlusion import optimize_occlusion
from foreground.planning import plan_foreground
from utils.io import append_jsonl, error_text, parse_size, read_json, write_json

LOGGER = logging.getLogger(__name__)


def load_references(config: Config) -> list[dict[str, Any]]:
    rows = read_json(config.references)
    references = []
    for row in rows:
        image = Path(row["image"])
        if not image.is_absolute():
            image = config.references.resolve().parent / image
        if not image.is_file():
            raise FileNotFoundError(image)
        references.append({**row, "image": str(image.resolve())})
    if not references:
        raise ValueError("Reference input is empty")
    return references


def prepare_run(config: Config, references: list[dict[str, Any]]) -> None:
    config.output.mkdir(parents=True, exist_ok=True)
    settings = {
        k: v
        for k, v in asdict(config).items()
        if k not in {"references", "output", "base_url", "concurrency", "request_timeout"}
    }
    # 输入或生成参数变化时拒绝复用旧输出目录，避免混入不同设置的结果。
    snapshot = {"configuration": settings, "references": references}
    path = config.output / "input_snapshot.json"
    if path.exists() and read_json(path) != snapshot:
        raise ValueError("Inputs or generation settings changed; use a new output directory")
    write_json(path, snapshot)


def synthesize_variant(
    spec: dict[str, Any],
    background: Path,
    items: list[dict[str, Any]],
    quality: dict[str, Any],
    variant_index: int,
    api: Any,
    config: Config,
) -> dict[str, Any]:
    image_id = f"{spec['spec_id']}__v{variant_index:03d}"
    annotation_path = config.output / "work/annotations" / f"{image_id}.json"
    if annotation_path.exists():
        return read_json(annotation_path)
    directory = config.output / "work/foreground" / image_id
    directory.mkdir(parents=True, exist_ok=True)
    # 每个变体都重新规划并生成前景，背景图层在变体之间共用。
    plan = plan_foreground(background, spec, api, directory, config.planner_model)
    plan = generate_assets(plan, api, directory, config.concurrency)
    size = parse_size(config.canvas_size)
    base, soft, hard = classify_layers(items)
    maps = foreground_region_maps(base, soft, hard, size)
    boxes = layout_boxes(plan["objects"], maps, size, variant_index)
    best = optimize_occlusion(plan, base, soft, hard, maps, size, boxes, max_candidates=config.max_occlusion_candidates)
    return save_annotation(image_id, best, spec, plan, quality, variant_index, config.output, size)


def synthesize_background(spec: dict[str, Any], api: Any, config: Config) -> dict[str, Any]:
    directory = config.output / "work/background" / spec["spec_id"]
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "spec.json", spec)
    image = generate_background(spec, api, directory)
    layers = plan_layers(image, api, directory, config.layer_model, config.max_layers)
    recomposed, records = extract_layers(image, layers, api, directory, config.concurrency)
    review = review_layers(image, recomposed, records, api, directory, config.layer_model)
    if not review["acceptable"]:
        return {"spec_id": spec["spec_id"], "status": "layer_review_rejected", "review": review, "annotations": []}
    background, items = clean_background(records, directory, parse_size(config.canvas_size))
    quality = screen_background(background, spec, api, directory, config.review_model)
    if quality["overall_score_1_to_5"] < config.background_score_threshold:
        return {
            "spec_id": spec["spec_id"],
            "status": "background_quality_rejected",
            "review": quality,
            "annotations": [],
        }
    rows, failures = [], []
    for index in range(config.variants):
        try:
            rows.append(synthesize_variant(spec, background, items, quality, index, api, config))
        except Exception as exc:
            failures.append({"variant_index": index, "error": error_text(exc, api)})
            LOGGER.exception("Foreground variant failed: %s/%s", spec["spec_id"], index)
    return {"spec_id": spec["spec_id"], "status": "accepted", "annotations": rows, "failures": failures}


def run(config: Config, api: Any = None) -> dict[str, Any]:
    started = time.monotonic()
    references = load_references(config)
    prepare_run(config, references)
    api = api if api is not None else ModelClient(config)
    specs, reference_failures = plan_specifications(references, api, config)
    results, generated, fallbacks = [], 0, 0
    # 每个背景完成后逐行追加；重跑时从缓存结果重新写出，不会重复。
    with (
        (config.output / "annotations.jsonl").open("w", encoding="utf-8") as annotation_file,
        (config.output / "background_results.jsonl").open("w", encoding="utf-8") as result_file,
        ThreadPoolExecutor(max_workers=config.concurrency) as executor,
    ):
        futures = {executor.submit(synthesize_background, spec, api, config): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "spec_id": spec["spec_id"],
                    "status": "failed",
                    "error": error_text(exc, api),
                    "annotations": [],
                }
                LOGGER.exception("Background synthesis failed: %s", spec["spec_id"])
            rows = result.pop("annotations")
            for row in rows:
                append_jsonl(annotation_file, row)
            append_jsonl(result_file, result)
            results.append(result)
            generated += len(rows)
            fallbacks += sum(bool(row["layout_quality"].get("fallback")) for row in rows)
            LOGGER.info("%s: %s", result["spec_id"], result["status"])
    summary = {
        "reference_count": len(references),
        "failed_references": len(reference_failures),
        "background_count": len(specs),
        "accepted_backgrounds": sum(r["status"] == "accepted" for r in results),
        "generated_designs": generated,
        "failed_backgrounds": sum(r["status"] == "failed" for r in results),
        "failed_variants": sum(len(r.get("failures", [])) for r in results),
        "fallback_layouts": fallbacks,
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    write_json(config.output / "summary.json", summary)
    write_json(config.output / "api_usage.json", getattr(api, "usage", []))
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = run(parse_args())
    LOGGER.info("Finished: %s", summary)
    return int(summary["failed_references"] + summary["failed_backgrounds"] + summary["failed_variants"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
