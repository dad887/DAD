import json
import unittest


from data.annotation.annotate import convert_response
from data.filtering.filter import overlap_components, select
from inference.common import load_config, parse_completion
from inference.evaluate import evaluate
from pathlib import Path
import yaml


class ReleaseTests(unittest.TestCase):
    def test_annotation_preserves_order_and_separates_types(self):
        boxes, labels = convert_response(
            json.dumps(
                {
                    "layers": [
                        {"type": "v", "bbox": [0.1, 0.2, 0.5, 0.7]},
                        {"type": "t", "bbox": [0.2, 0.3, 0.4, 0.5]},
                    ]
                }
            ),
            200,
            100,
        )
        self.assertEqual(boxes, [[20, 20, 100, 70], [40, 30, 80, 50]])
        self.assertEqual(labels, ["visual", "text"])

    def test_annotation_rejects_invalid_box(self):
        with self.assertRaises(ValueError):
            convert_response('{"layers":[{"type":"v","bbox":[0.5,0,0.2,1]}]}', 100, 100)

    def test_eos_does_not_drop_last_detection(self):
        boxes, labels = parse_completion("0,0,100,100,t;200,200,400,400,v<|im_end|>")
        self.assertEqual(len(boxes), 2)
        self.assertEqual(labels.tolist(), ["text", "visual"])

    def test_hungarian_penalizes_duplicate_predictions(self):
        row = {
            "id": "a",
            "width": 1000,
            "height": 1000,
            "boxes": [[0, 0, 100, 100]],
            "types": ["visual"],
        }
        config = load_config()
        perfect = evaluate([row], [{"id": "a", "completion": "0,0,100,100,v"}], config)
        duplicate = evaluate(
            [row], [{"id": "a", "completion": "0,0,100,100,v;0,0,100,100,v;"}], config
        )
        self.assertAlmostEqual(perfect["iou_weighted_f1"], 100, places=5)
        self.assertAlmostEqual(
            duplicate["iou_weighted_f1"], 100 * (0.8 * 2 / 3 + 0.2), places=5
        )

    def test_eval_requires_exact_coverage(self):
        rows = [{"id": "missing"}]
        with self.assertRaises(ValueError):
            evaluate(rows, [], load_config())
        with self.assertRaises(ValueError):
            evaluate([], [], load_config())

    def test_containment_and_touching_do_not_form_components(self):
        boxes = [[0, 0, 100, 100], [20, 20, 30, 30], [100, 0, 110, 10]]
        self.assertEqual(sorted(map(len, overlap_components(boxes))), [1, 1, 1])

    def test_overlap_connections_are_transitive(self):
        boxes = [[0, 0, 20, 20], [10, 0, 30, 20], [20, 0, 40, 20]]
        self.assertEqual(sorted(map(len, overlap_components(boxes))), [3])

    def test_small_recall_boundary_and_type_matching(self):
        config = yaml.safe_load(
            (Path(__file__).parents[1] / "data/filtering/config.yaml").read_text()
        )
        boxes = [[i * 20, 0, i * 20 + 10, 10] for i in range(5)]
        row = {
            "id": "a",
            "width": 100,
            "height": 100,
            "boxes": boxes,
            "types": ["visual"] * 5,
        }
        self.assertTrue(select(row, {"boxes": [], "types": []}, config)[0])
        self.assertTrue(
            select(row, {"boxes": boxes[:1], "types": ["visual"]}, config)[0]
        )
        self.assertFalse(
            select(row, {"boxes": boxes[:2], "types": ["visual"] * 2}, config)[0]
        )
        self.assertTrue(
            select(row, {"boxes": boxes[:2], "types": ["text"] * 2}, config)[0]
        )

    def test_dense_noncontained_overlap_is_filtered(self):
        config = yaml.safe_load(
            (Path(__file__).parents[1] / "data/filtering/config.yaml").read_text()
        )
        boxes = [[i, 0, 20 + i, 20] for i in range(10)]
        row = {
            "id": "a",
            "width": 100,
            "height": 100,
            "boxes": boxes,
            "types": ["visual"] * 10,
        }
        self.assertEqual(
            select(row, {"boxes": [], "types": []}, config)[1], "over_split"
        )


if __name__ == "__main__":
    unittest.main()
