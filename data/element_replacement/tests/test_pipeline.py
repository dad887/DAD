from __future__ import annotations

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from background.screening import classify_layers, clean_background
from background.specifications import validate_designs
from config import Config
from foreground.metrics import analyze_stack_fast
from run_pipeline import run
from utils.geometry import content_bbox_from_mask
from utils.io import read_json, write_json
from utils.masking import remove_matte_background, remove_white_background


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def design() -> dict:
    return {
        "theme_category": "education",
        "theme": "a playful learning event",
        "title_text": "LEARN & PLAY",
        "title_position": "top_compact",
        "style_prompt": "clean flat vector illustration",
        "palette_hex": ["#BDEBFF", "#E43D32", "#FDBE32", "#74BC44", "#223344"],
        "layout_description": "compact top headline and open central area",
        "title_bbox_norm": [0.1, 0.05, 0.8, 0.1],
        "safe_zones_norm": [[0.1, 0.3, 0.8, 0.5]],
        "decor_layers": ["one broad green hill across the bottom"],
    }


class FakeModels:
    def __init__(self, reject_layers: bool = False, empty_layer: bool = False, background_score: int = 5):
        self.calls = []
        self.usage = []
        self.reject_layers = reject_layers
        self.empty_layer = empty_layer
        self.background_score = background_score

    def json(self, prompt: str, images: list[Path], model: str):
        self.calls.append((model, prompt))
        if "PREASSIGNED MODES" in prompt:
            return [design() for _ in range(3)]
        if "planning layer decomposition" in prompt:
            return {
                "layers": [
                    {
                        "layer_id": "base",
                        "role": "base",
                        "element_type": "visual",
                        "z_order": 0,
                        "description_en": "flat blue canvas base",
                        "extraction_prompt_en": "only the blue base",
                    },
                    {
                        "layer_id": "headline",
                        "role": "target",
                        "element_type": "text",
                        "z_order": 2,
                        "description_en": "red title text",
                        "extraction_prompt_en": "only the red headline",
                    },
                    {
                        "layer_id": "hill",
                        "role": "target",
                        "element_type": "visual",
                        "z_order": 1,
                        "description_en": "green decorative hill",
                        "extraction_prompt_en": "only the green hill",
                    },
                ]
            }
        if "Compare image A" in prompt:
            return {"acceptable": not self.reject_layers, "main_differences": []}
        if "strict quality reviewer" in prompt:
            return {"overall_score_1_to_5": self.background_score, "reason": "Mock background quality decision"}
        if "planning foreground cutout" in prompt:
            return {
                "motif": "two learning props",
                "style_directive": "flat clean vector shapes",
                "group_layout": {"arrangement": "large focal object and smaller support"},
                "candidate_assets": ["single book", "single pencil", "single globe", "single student", "single ruler"],
                "selected_assets": [
                    {"role": "focal", "asset_label": "single red book", "scale": "large"},
                    {"role": "supporting", "asset_label": "single green globe", "scale": "medium"},
                ],
            }
        raise AssertionError("Unrecognized stage prompt")

    def generate(self, prompt: str, path: Path, size: str) -> Path:
        if path.exists():
            return path
        self.calls.append(("generate", prompt))
        width, height = map(int, size.split("x"))
        path.parent.mkdir(parents=True, exist_ok=True)
        if "isolated foreground cutout" in prompt:
            image = Image.new("RGB", (width, height), "#dddddd")
            draw = ImageDraw.Draw(image)
            draw.rectangle(
                (width // 3, height // 4, 2 * width // 3, 3 * height // 4),
                fill="#c32f2f" if "red book" in prompt else "#24884d",
            )
        else:
            image = Image.new("RGB", (width, height), "#BDEBFF")
        image.save(path)
        return path

    def extract(self, prompt: str, source: Path, path: Path, size: str) -> Path:
        if path.exists():
            return path
        self.calls.append(("extract", prompt))
        width, height = map(int, size.split("x"))
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (width, height), "#BDEBFF" if "blue base" in prompt else "white")
        draw = ImageDraw.Draw(image)
        if "red headline" in prompt and not self.empty_layer:
            draw.rectangle((width // 5, height // 16, 4 * width // 5, height // 8), fill="#E43D32")
        if "green hill" in prompt:
            draw.rectangle((0, 7 * height // 8, width - 1, height - 1), fill="#74BC44")
        image.save(path)
        return path


class PipelineTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "tmp/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temporary.name)
        Image.new("RGB", (256, 384), "#BDEBFF").save(self.root / "reference.png")
        references = [
            {
                "reference_id": "education",
                "source_id": "test",
                "image": "reference.png",
                "annotations": {"caption": "A blue and green education poster with a red headline"},
            }
        ]
        write_json(self.root / "references.json", references)
        self.config = Config(
            references=self.root / "references.json",
            output=self.root / "output",
            canvas_size="256x384",
            variants=1,
            concurrency=2,
            max_occlusion_candidates=12,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_pipeline_and_resume(self):
        api = FakeModels()
        summary = run(self.config, api)
        self.assertEqual(summary["generated_designs"], 3)
        self.assertEqual(summary["failed_backgrounds"], 0)
        self.assertEqual(summary["failed_variants"], 0)
        rows = read_jsonl(self.config.output / "annotations.jsonl")
        for row in rows:
            self.assertTrue(row["recompose_reproducibility"]["exact_match"])
            self.assertIn("text", row["types"])
            self.assertEqual(len(row["boxes"]), len(row["types"]))
            self.assertEqual(len(row["boxes"]), len(row["visible_bboxes"]))
            self.assertEqual(len(row["boxes"]), len(row["target_layer_ids"]))
            self.assertTrue(any(layer["layer_group"] == "shadow" for layer in row["layers"]))
            self.assertFalse(any("rgba_path" in obj for obj in row["foreground_plan"]["objects"]))
            for layer in row["layers"]:
                self.assertTrue((self.config.output / layer["layer_path"]).is_file())
        self.assertTrue(all(not re.search("[\u4e00-\u9fff]", prompt) for _, prompt in api.calls))
        calls = len(api.calls)
        resumed = run(self.config, api)
        self.assertEqual(resumed["generated_designs"], 3)
        self.assertEqual(len(api.calls), calls)
        self.assertEqual(len(read_jsonl(self.config.output / "annotations.jsonl")), 3)

    def test_rejected_layers_never_generate_foreground(self):
        api = FakeModels(reject_layers=True)
        summary = run(self.config, api)
        self.assertEqual(summary["accepted_backgrounds"], 0)
        self.assertEqual(summary["generated_designs"], 0)
        self.assertFalse(any("planning foreground cutout" in prompt for _, prompt in api.calls))

    def test_empty_required_layer_is_not_published(self):
        with self.assertLogs("run_pipeline", level="ERROR"):
            summary = run(self.config, FakeModels(empty_layer=True))
        self.assertEqual(summary["failed_backgrounds"], 3)
        self.assertEqual(summary["generated_designs"], 0)
        self.assertEqual(read_jsonl(self.config.output / "annotations.jsonl"), [])

    def test_background_quality_rejection_stops_foreground(self):
        api = FakeModels(background_score=4)
        summary = run(self.config, api)
        self.assertEqual(summary["accepted_backgrounds"], 0)
        self.assertEqual(summary["generated_designs"], 0)
        self.assertFalse(any("planning foreground cutout" in prompt for _, prompt in api.calls))

    def test_texture_is_not_classified_as_text(self):
        item = {"record": {"role": "target", "element_type": "visual", "description_en": "textured paper decoration"}}
        _, soft, hard = classify_layers([item])
        self.assertEqual(len(soft), 1)
        self.assertEqual(hard, [])
        item["record"]["description_en"] = "title label panel"
        _, soft, hard = classify_layers([item])
        self.assertEqual(soft, [])
        self.assertEqual(len(hard), 1)

    def test_input_changes_require_new_output(self):
        run(self.config, FakeModels(reject_layers=True))
        refs = read_json(self.config.references)
        refs[0]["annotations"]["caption"] = "A changed reference theme"
        write_json(self.config.references, refs)
        with self.assertRaises(ValueError):
            run(self.config, FakeModels())

    def test_planner_schema_is_exactly_three_designs(self):
        validate_designs([design(), design(), design()])
        with self.assertRaises(ValueError):
            validate_designs([design()])
        wrong = copy.deepcopy(design())
        wrong["image_prompt"] = "Unexpected model-written prompt"
        with self.assertRaises(ValueError):
            validate_designs([wrong, design(), design()])

    def test_white_interior_is_preserved(self):
        image = Image.new("RGBA", (100, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((15, 15, 85, 85), outline="black", width=8)
        result = remove_white_background(image)
        self.assertEqual(result.getpixel((0, 0))[3], 0)
        self.assertEqual(result.getpixel((50, 50))[3], 255)

    def test_white_layer_survives_gray_matte_and_screening(self):
        raw = Image.new("RGBA", (100, 100), "#808080")
        ImageDraw.Draw(raw).ellipse((20, 20, 80, 80), fill="white")
        rgba = remove_matte_background(raw, (128, 128, 128))
        self.assertEqual(rgba.getpixel((0, 0))[3], 0)
        self.assertEqual(rgba.getpixel((50, 50)), (255, 255, 255, 255))
        base_path = self.root / "base.png"
        cloud_path = self.root / "cloud.png"
        raw_path = self.root / "raw_cloud.png"
        Image.new("RGBA", (100, 100), "#BDEBFF").save(base_path)
        raw.save(raw_path)
        rgba.save(cloud_path)
        records = [
            {"generated": True, "role": "base", "z_order": 0, "layer_id": "base", "rgba_path": str(base_path)},
            {
                "generated": True,
                "role": "target",
                "z_order": 1,
                "layer_id": "cloud",
                "element_type": "visual",
                "description_en": "white cloud",
                "raw_path": str(raw_path),
                "rgba_path": str(cloud_path),
            },
        ]
        path, _ = clean_background(records, self.root, (100, 100))
        with Image.open(path) as cleaned:
            self.assertEqual(cleaned.getpixel((0, 0)), (189, 235, 255))
            self.assertEqual(cleaned.getpixel((50, 50)), (255, 255, 255))

    def test_pixel_and_box_occlusion_differ(self):
        lower = np.zeros((16, 16), dtype=bool)
        lower[2:12, 2:12] = True
        upper = np.zeros_like(lower)
        upper[2:12, 5:9] = True

        def layer(name, alpha):
            return {
                "kind": "fg",
                "layer_group": "fg",
                "object_id": name,
                "role": "supporting",
                "alpha": alpha,
                "alpha_area": int(alpha.sum()),
                "content_bbox": content_bbox_from_mask(alpha),
            }

        stats, events = analyze_stack_fast(
            [layer("lower", lower), layer("upper", upper)], (16, 16), target_groups={"fg"}
        )
        self.assertAlmostEqual(stats[0]["pixel_occlusion_ratio"], 0.4)
        self.assertEqual(stats[0]["box_occlusion_ratio"], 0)
        self.assertEqual(stats[0]["visible_bbox"], [2, 2, 12, 12])
        self.assertEqual(events[0]["target"], "lower")
        upper[:] = False
        upper[2:12, 8:12] = True
        stats, _ = analyze_stack_fast([layer("lower", lower), layer("upper", upper)], (16, 16), target_groups={"fg"})
        self.assertAlmostEqual(stats[0]["box_occlusion_ratio"], 0.4)
        self.assertEqual(stats[0]["visible_bbox"], [2, 2, 8, 12])


if __name__ == "__main__":
    unittest.main()
