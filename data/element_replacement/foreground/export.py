from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from foreground.metrics import target_id
from utils.geometry import bbox_area, bbox_to_int, content_bbox_from_mask
from utils.image_ops import add_shadow
from utils.io import slug, write_json
from utils.masking import ALPHA_THRESH


def make_shadow_layer(item: dict[str, Any], size: tuple[int, int]) -> dict[str, Any] | None:
    # 阴影单独成层，不作为检测目标，也不参与遮挡统计。
    if item.get("kind") != "fg":
        return None
    bbox = bbox_to_int(item.get("content_bbox") or content_bbox_from_mask(item["alpha"]))
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    full = Image.new("RGBA", size, (0, 0, 0, 0))
    crop = item["image"].convert("RGBA").crop(bbox)
    add_shadow(full, crop, (bbox[0], bbox[1]))
    alpha = np.asarray(full.getchannel("A")) > ALPHA_THRESH
    if not alpha.any():
        return None
    oid = target_id(item)
    return {
        "kind": "shadow",
        "layer_group": "shadow",
        "layer_id": f"{oid}_shadow",
        "object_id": f"{oid}_shadow",
        "source_object_id": oid,
        "image": full,
        "alpha": alpha,
        "alpha_area": int(alpha.sum()),
        "content_bbox": content_bbox_from_mask(alpha),
    }


def materialize_render_stack(stack: list[dict[str, Any]], size: tuple[int, int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in stack:
        shadow = make_shadow_layer(item, size)
        if shadow is not None:
            out.append(shadow)
        out.append(item)
    return out


def alpha_composite_layer_images(layers: list[dict[str, Any]], size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    for layer in layers:
        canvas.alpha_composite(layer["image"].convert("RGBA"))
    return canvas.convert("RGB")


def verify_recompose_from_saved_layers(
    final_path: Path, layer_records: list[dict[str, Any]], out_root: Path, size: tuple[int, int]
) -> dict[str, Any]:
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    for rec in sorted(layer_records, key=lambda r: int(r["z_index"])):
        canvas.alpha_composite(Image.open(out_root / rec["layer_path"]).convert("RGBA"))
    recomposed = canvas.convert("RGB")
    final = Image.open(final_path).convert("RGB")
    a = np.asarray(final, dtype=np.int16)
    b = np.asarray(recomposed, dtype=np.int16)
    diff = np.abs(a - b)
    return {"exact_match": bool(diff.max() == 0), "max_abs_diff": int(diff.max()), "mean_abs_diff": float(diff.mean())}


def render_final_layers(
    image_id: str, stack: list[dict[str, Any]], out_root: Path, size: tuple[int, int]
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    layer_dir = out_root / "layers" / image_id
    layer_dir.mkdir(parents=True, exist_ok=True)
    layer_records: list[dict[str, Any]] = []
    render_stack = materialize_render_stack(stack, size)
    for z, layer in enumerate(render_stack):
        lid = target_id(layer)
        path = layer_dir / f"{z:03d}_{slug(lid)}.png"
        layer["image"].save(path)
        is_shadow = layer.get("layer_group") == "shadow"
        layer_records.append(
            {
                "layer_id": lid,
                "layer_path": str(path.relative_to(out_root)),
                "layer_group": layer.get("layer_group"),
                "source": "shadow"
                if is_shadow
                else "generated_foreground"
                if layer.get("kind") == "fg"
                else "generated_background",
                "z_index": z,
                "bbox": layer.get("content_bbox") or content_bbox_from_mask(layer["alpha"]),
                "is_detection_target": layer.get("layer_group") not in {"base", "shadow"},
                "role": layer.get("role"),
                "asset_label": layer.get("asset_label"),
                "source_object_id": layer.get("source_object_id"),
                "element_type": layer.get("record", {}).get("element_type", "visual"),
            }
        )
    final = alpha_composite_layer_images(render_stack, size)
    image_path = out_root / "images" / f"{image_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(image_path)
    recompose_check = verify_recompose_from_saved_layers(image_path, layer_records, out_root, size)
    return (image_path, layer_records, recompose_check)


def save_annotation(
    image_id: str,
    best: dict[str, Any],
    spec: dict[str, Any],
    plan: dict[str, Any],
    background_quality: dict[str, Any],
    variant_index: int,
    out_root: Path,
    size: tuple[int, int],
) -> dict[str, Any]:
    image_path, records, check = render_final_layers(image_id, best["stack"], out_root, size)
    # 保存的图层必须逐像素还原成图，否则不导出。
    if not check["exact_match"]:
        raise ValueError("Saved layers do not reproduce the final image")
    stat_by_id = {s["target_id"]: s for s in best["all_target_stats"]}
    boxes, visible_boxes, types, target_ids = [], [], [], []
    enriched = []
    for record in records:
        stats = stat_by_id.get(record["layer_id"])
        if record["is_detection_target"] and stats is not None:
            target_ids.append(record["layer_id"])
            boxes.append(stats["bbox"])
            visible_boxes.append(stats["visible_bbox"])
            types.append(record["element_type"])
            record = {
                **record,
                "annotation_index": len(boxes) - 1,
                **{
                    k: stats[k]
                    for k in [
                        "visible_bbox",
                        "mask_area",
                        "visible_mask_area",
                        "pixel_occlusion_ratio",
                        "box_occlusion_ratio",
                        "visible_ratio",
                    ]
                },
            }
        enriched.append(record)
    public_plan = {
        **plan,
        "objects": [{k: v for k, v in obj.items() if k not in {"raw_path", "rgba_path"}} for obj in plan["objects"]],
    }
    occluded_area = sum(max(0, bbox_area(box) - bbox_area(visible)) for box, visible in zip(boxes, visible_boxes))
    row = {
        "image": str(image_path.relative_to(out_root)),
        "primary_key": image_id,
        "page_index": 0,
        "width": size[0],
        "height": size[1],
        "boxes": boxes,
        "types": types,
        "visible_bboxes": visible_boxes,
        "subdir": spec["spec_id"],
        "n_layers": len(boxes),
        "target_layer_ids": target_ids,
        "total_occluded_bbox_area": int(round(occluded_area)),
        "occlusion_ratio": occluded_area / max(1, size[0] * size[1]),
        "source_sample": spec["spec_id"],
        "reference_id": spec["reference_id"],
        "variant_index": variant_index,
        "design_specification": spec,
        "background_quality_review": background_quality,
        "foreground_plan": public_plan,
        "layers": enriched,
        "z_order": [r["layer_id"] for r in records],
        "recompose_reproducibility": check,
        "occlusion_stats": best["all_target_stats"],
        "occlusion_events": best["occlusion_events"],
        "layout_quality": best["quality_gate"],
    }
    write_json(out_root / "work/annotations" / f"{image_id}.json", row)
    return row
