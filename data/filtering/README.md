# Design selection

This implements the data-selection procedure in Appendix B.

1. Keep designs with 5–10 small elements, each occupying 1%–5% of the image area, whose preliminary-detector recall is at most 20%.
2. Connect intersecting boxes when neither contains the other. Reject designs with a connected component of at least 10 boxes whose summed box area divided by its enclosing-box area exceeds 2.0.

Use candidate designs that were not used to train the preliminary detector. A supplied exclusion list can enforce this explicitly.

```bash
python -m data.filtering.filter \
  --annotations candidates.jsonl \
  --predictions preliminary_predictions.jsonl \
  --exclude-ids preliminary_training_ids.txt \
  --output-dir selected \
  --test-size 5000
```

Both JSONL inputs are keyed by `id`. Annotations contain pixel-space `boxes`, `types`, `width`, and `height`. Predictions contain pixel-space `boxes` and `types`; an image with no detections must have an explicit record with empty lists. Types are `text`/`visual` or `t`/`v`.

Defaults in [config.yaml](config.yaml) use type-aware one-to-one matching at IoU ≥ 0.5 for small-element recall. Area, count, and recall interval endpoints are inclusive; the overlap fill-ratio comparison is strictly greater than 2.0. Containment and edge-only contact do not connect boxes.

Outputs include `selected.jsonl` and a summary of selection counts and settings. Providing `--test-size` also produces deterministic `rl.jsonl` and `test.jsonl` files. Omit that option to perform selection only. The published public-experiment splits are already fixed and do not need this filter to be rerun.
