"""Evaluate predictions against the fixed public split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from elerpo.stepwise.iou_utils import compute_hungarian_iou_f1_reward
from .common import (
    CONFIG_PATH,
    TYPE_MAP,
    load_config,
    normalized_boxes,
    parse_completion,
    read_jsonl,
)


def evaluate(annotations, predictions, config):
    ids = [r["id"] for r in annotations]
    if len(set(ids)) != len(ids):
        raise ValueError("Annotation record IDs must be unique")
    by_id = {}
    for record in predictions:
        if record["id"] in by_id:
            raise ValueError("Duplicate prediction ID: " + record["id"])
        by_id[record["id"]] = record
    missing, extra = set(ids) - by_id.keys(), by_id.keys() - set(ids)
    if missing or extra:
        raise ValueError(
            f"Prediction coverage mismatch: {len(missing)} missing, {len(extra)} unexpected"
        )
    if not annotations:
        raise ValueError("No annotations")
    totals = {"text": 0.0, "visual": 0.0}
    for row in annotations:
        if len(row["boxes"]) != len(row["types"]):
            raise ValueError("Mismatched boxes and types: " + row["id"])
        gt = normalized_boxes(row, config["max_image_size"])
        labels = np.asarray([TYPE_MAP[t] for t in row["types"]], dtype=object)
        pred, pred_types = parse_completion(by_id[row["id"]]["completion"])
        for kind in totals:
            totals[kind] += compute_hungarian_iou_f1_reward(
                gt[labels == kind], pred[pred_types == kind]
            )
    means = {k: v / len(annotations) for k, v in totals.items()}
    score = (
        config["visual_weight"] * means["visual"]
        + config["text_weight"] * means["text"]
    )
    return {
        "num_examples": len(annotations),
        "iou_weighted_f1": 100 * score,
        "visual_iou_f1": 100 * means["visual"],
        "text_iou_f1": 100 * means["text"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = evaluate(
        read_jsonl(args.annotations),
        read_jsonl(args.predictions),
        load_config(args.config),
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
