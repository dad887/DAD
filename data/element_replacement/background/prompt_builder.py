from __future__ import annotations

import json
import random
from typing import Any


def prompt_from_spec(spec: dict[str, Any]) -> str:
    palette = ", ".join(spec["palette_hex"])
    decor = ", ".join(spec["decor_layers"])
    safe_zones = json.dumps(spec["safe_zones_norm"])
    title_box = json.dumps(spec["title_bbox_norm"])
    title_text = spec["title_text"].strip()
    shape_rules = " ".join(spec["layer_constraints"]["shape_rules"])
    title_instruction = (
        f'Use exact main headline text "{title_text}" inside the requested title area.'
        if spec["title_position"] != "label_or_no_title"
        else f'Use "{title_text}" only as a tiny label if helpful; no dominant headline.'
    )
    # 设计字段填入公共模板，再按模式追加不同的分层约束。
    common = f"Create one original high-quality graphic design background/base image for a public synthetic dataset. This is pure text-to-image generation; do not imitate any specific existing template. Canvas size request: {spec['image_size']}. Theme: {spec['theme']}. Theme category: {spec['theme_category']}. Style: {spec['style_prompt']}. Palette: {palette}. Layout: {spec['layout_description']}. Title area normalized [x,y,w,h]: {title_box}. {title_instruction} Foreground-safe open zones normalized [x,y,w,h]: {safe_zones}. Include only these clearly separable decorative graphic ingredients: {decor}. Make the composition visibly different from a standard top-title centered poster; respect the requested title position. No people, no animals, no standalone foreground objects, no real brand names, no logos, no watermark, no QR code, no small body text, no copyrighted characters. Do not recreate any known template; create new public-safe pixels."
    if spec["generation_mode"] == "visual_rich":
        return (
            common
            + " Generation mode: visual_rich. "
            + "CRITICAL DOWNSTREAM TASK: an automatic layer-splitting pipeline will split this image into important editable graphic layers, but may intentionally omit tiny grain, faint texture, or minor ornaments and may mark complex unsuitable groups as group_skip. "
            + "Use a richer graphic-design look such as organized collage, scrapbook, textured poster, soft dimensional abstract forms, layered paper, stamps, ribbons, or halftone corners. "
            + "Keep the title, major panels, major bands, and major decorative groups readable as separate visual groups. "
            + "Do not overfill the canvas; use roughly 8 to 12 visible visual groups, with optional tiny texture treated as background detail. "
            + f"Mode-specific rules: {shape_rules} "
            + "Allow controlled overlap and tactile texture, but avoid brand-like marks, copyrighted imagery, realistic objects, dense tiny icons, and chaotic confetti."
        )
    return (
        common
        + " Generation mode: layer_friendly. "
        + "CRITICAL DOWNSTREAM TASK: this image will be automatically split into graphic layers by an automatic layer-splitting pipeline, then recomposed locally to create bbox and mask annotations. "
        + "Therefore the design must be intentionally layer-friendly, simple, and easy to segment. "
        + "Aim for 5 to 8 visible graphic layers total: base texture, one clean title layer, 2 to 4 large decorative shapes, one border/line layer, and at most one sparse accent cluster. "
        + f"Layer-friendly rules: {shape_rules} "
        + "Use crisp readable typography, abstract decoration, broad panels, waves, blobs, border lines, one brush mark, and subtle paper texture. "
        + "Avoid visual clutter. Avoid overlapping scraps. Avoid realistic tape, paper clips, pencils, paper airplanes, blocks, plant pots, stickers, and 3D props. "
        + "Avoid dense dot fields, tiny repeated icons, complex shadows, and many independent objects."
    )


LAYER_FRIENDLY_CONSTRAINTS = {
    "generation_task": "This background will be processed by an automatic layer-splitting pipeline. The image must be easy to decompose into clean RGBA layers.",
    "target_total_layers": "5 to 8 visible layers including base, title, panels, waves/blobs, border/line, and at most one small accent cluster",
    "target_train_layers": "4 to 7 clear target layers after excluding the base texture",
    "shape_rules": [
        "Prefer large simple shapes over many small objects.",
        "Keep each decorative element spatially separated from other elements.",
        "Use flat or lightly textured graphics, not realistic objects.",
        "Use minimal shadows; avoid occluding one decorative layer with another.",
        "Keep dot clusters sparse and localized.",
        "Keep title text separable from its label/background panel.",
    ],
}

VISUAL_RICH_CONSTRAINTS = {
    "generation_task": "This background will still be processed by the automatic layer-splitting pipeline. The image may be richer, but important visual groups should remain recognizable.",
    "target_total_layers": "8 to 12 visible visual groups is acceptable; tiny texture and minor ornaments may be omitted by the layer-splitting pipeline",
    "target_train_layers": "4 to 8 important target layers after excluding base texture and skipped complex groups",
    "shape_rules": [
        "Allow collage, scrapbook, textured poster, or soft dimensional styling, but avoid uncontrolled clutter.",
        "Keep the title, major panels, major bands, and major decorative groups readable as separate objects.",
        "Small grain, faint texture, and tiny optional ornaments are allowed because the pipeline can record them as omitted_elements.",
        "Complex decorative regions should look like coherent groups that the pipeline may mark group_skip instead of atomic targets.",
        "Do not include brands, watermarks, real people, or copyrighted visual characters.",
    ],
}


def constraints_for_mode(generation_mode: str) -> dict[str, Any]:
    if generation_mode == "visual_rich":
        return VISUAL_RICH_CONSTRAINTS
    return LAYER_FRIENDLY_CONSTRAINTS


def mode_sequence(count: int, layer_friendly_ratio: float, rng: random.Random) -> list[str]:
    friendly = round(count * layer_friendly_ratio)
    seq = ["layer_friendly"] * friendly + ["visual_rich"] * (count - friendly)
    rng.shuffle(seq)
    return seq
