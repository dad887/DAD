from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    references: Path
    output: Path
    base_url: str | None = None
    planner_model: str = "gpt-5.5"
    layer_model: str = "gpt-5-mini"
    review_model: str = "gpt-5.5"
    image_model: str = "gpt-image-2"
    image_quality: str = "low"
    canvas_size: str = "1024x1536"
    concurrency: int = 10
    variants: int = 30
    seed: int = 20261008
    layer_friendly_ratio: float = 0.8
    max_layers: int = 12
    max_occlusion_candidates: int = 240
    background_score_threshold: int = 5
    request_timeout: int = 300


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Reference-driven layered graphic synthesis")
    parser.add_argument(
        "--references", type=Path, required=True, help="Reference images and annotations in one JSON array"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    parser.add_argument("--planner-model", default=Config.planner_model)
    parser.add_argument("--layer-model", default=Config.layer_model)
    parser.add_argument("--review-model", default=Config.review_model)
    parser.add_argument("--image-model", default=Config.image_model)
    parser.add_argument("--image-quality", choices=["low", "medium", "high"], default=Config.image_quality)
    parser.add_argument("--canvas-size", choices=["1024x1536", "1536x1024", "1024x1024"], default=Config.canvas_size)
    parser.add_argument("--concurrency", type=int, default=Config.concurrency)
    parser.add_argument("--variants", type=int, default=Config.variants)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--layer-friendly-ratio", type=float, default=Config.layer_friendly_ratio)
    parser.add_argument("--max-layers", type=int, default=Config.max_layers)
    parser.add_argument("--max-occlusion-candidates", type=int, default=Config.max_occlusion_candidates)
    parser.add_argument(
        "--background-score-threshold", type=int, choices=range(1, 6), default=Config.background_score_threshold
    )
    parser.add_argument("--request-timeout", type=int, default=Config.request_timeout)
    args = parser.parse_args()
    for name in ["concurrency", "variants", "max_layers", "max_occlusion_candidates", "request_timeout"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.layer_friendly_ratio <= 1:
        parser.error("--layer-friendly-ratio must be between 0 and 1")
    return Config(**vars(args))
