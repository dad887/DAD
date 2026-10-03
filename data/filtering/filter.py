"""Select designs using the small-element and overlap rules in Appendix B."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
from scipy.optimize import linear_sum_assignment
import yaml

from elerpo.stepwise.iou_utils import compute_iou_matrix
from inference.common import TYPE_MAP, read_jsonl


def overlap_components(boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            intersects = min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(
                a[1], b[1]
            )
            contains_a = a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]
            contains_b = b[0] <= a[0] and b[1] <= a[1] and b[2] >= a[2] and b[3] >= a[3]
            if intersects and not contains_a and not contains_b:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def validate_record(row):
    boxes = np.asarray(row["boxes"], dtype=float).reshape(-1, 4)
    if row["width"] <= 0 or row["height"] <= 0 or len(boxes) != len(row["types"]):
        raise ValueError("Invalid dimensions or labels: " + row["id"])
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Non-finite or non-positive box: " + row["id"])
    for label in row["types"]:
        if label not in TYPE_MAP:
            raise ValueError("Unknown type in " + row["id"])
    return boxes


def select(row, prediction, config):
    boxes = validate_record(row)
    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    relative = areas / (row["width"] * row["height"])
    small = (relative >= config["small_area_min"]) & (
        relative <= config["small_area_max"]
    )
    count = int(small.sum())
    if not config["small_count_min"] <= count <= config["small_count_max"]:
        return False, "small_count", {"small_count": count}
    pred = np.asarray(prediction["boxes"], dtype=float).reshape(-1, 4)
    pred_types = prediction["types"]
    if len(pred) != len(pred_types) or not np.isfinite(pred).all():
        raise ValueError("Invalid preliminary predictions: " + row["id"])
    ious = compute_iou_matrix(boxes[small].astype(np.float32), pred.astype(np.float32))
    eligible = ious >= config["recall_iou_threshold"]
    if config["match_types"]:
        gt_types = np.asarray([TYPE_MAP[t] for t in row["types"]], dtype=object)[small]
        pt = np.asarray([TYPE_MAP[t] for t in pred_types], dtype=object)
        eligible &= gt_types[:, None] == pt[None, :]
    # Maximize the number of valid matches, then their total overlap.
    gi, pi = linear_sum_assignment(-(eligible * (count + 1) + ious * eligible))
    recall = float(eligible[gi, pi].sum()) / count
    if not config["small_recall_min"] <= recall <= config["small_recall_max"]:
        return False, "small_recall", {"small_count": count, "small_recall": recall}
    for component in overlap_components(boxes):
        if len(component) < config["overlap_component_min_boxes"]:
            continue
        group = boxes[component]
        enclosing = np.prod(group[:, 2:].max(axis=0) - group[:, :2].min(axis=0))
        ratio = float(areas[component].sum() / enclosing)
        if ratio > config["overlap_fill_ratio"]:
            return (
                False,
                "over_split",
                {"component_boxes": len(component), "fill_ratio": ratio},
            )
    return True, "selected", {"small_count": count, "small_recall": recall}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Pixel-space preliminary detector boxes and types, keyed by id",
    )
    p.add_argument(
        "--exclude-ids", type=Path, help="One preliminary-training sample ID per line"
    )
    p.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--test-size",
        type=int,
        help="Optionally split selected designs into RL and test sets",
    )
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text())
    rows = read_jsonl(args.annotations)
    predictions = read_jsonl(args.predictions)
    if len({x["id"] for x in rows}) != len(rows) or len(
        {x["id"] for x in predictions}
    ) != len(predictions):
        p.error("Input IDs must be unique")
    by_id = {x["id"]: x for x in predictions}
    excluded = (
        set(args.exclude_ids.read_text().splitlines()) if args.exclude_ids else set()
    )
    selected = []
    counts = Counter()
    for row in rows:
        if row["id"] in excluded:
            counts["preliminary_training"] += 1
            continue
        if row["id"] not in by_id:
            raise ValueError("Missing prediction: " + row["id"])
        keep, reason, _ = select(row, by_id[row["id"]], config)
        counts[reason] += 1
        if keep:
            selected.append(row)
    if args.test_size is not None and not 0 < args.test_size < len(selected):
        p.error("--test-size must leave at least one selected record for RL")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def write(name, records):
        with (args.output_dir / name).open("w") as f:
            for row in records:
                f.write(json.dumps(row) + "\n")

    write("selected.jsonl", selected)
    if args.test_size is not None:
        split_rows = sorted(selected, key=lambda x: x["id"])
        random.Random(config["split_seed"]).shuffle(split_rows)
        write("test.jsonl", split_rows[: args.test_size])
        write("rl.jsonl", split_rows[args.test_size :])
    summary = {"input_designs": len(rows), "counts": dict(counts), "parameters": config}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
