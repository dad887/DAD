# Inference and evaluation

Run commands from the repository root after installing `requirements.txt`.

## Fixed public test split

```bash
python -m inference.predict --variant sft --output outputs/sft.jsonl
python -m inference.predict --variant grpo --output outputs/grpo.jsonl
python -m inference.predict --variant elerpo --output outputs/elerpo.jsonl

python -m inference.evaluate \
  --annotations public_data/annotations/test.jsonl \
  --predictions outputs/elerpo.jsonl \
  --output outputs/elerpo_metrics.json
```

The first inference call prepares the data under `public_data/`. Downloads are cached. Each selected image is checked against its SHA-256 digest. All 5,000 test records are processed, including a final partial batch. Evaluation requires exactly one prediction for every annotation ID and rejects missing, duplicate, or unexpected records.

The evaluator reports mean IoU-weighted F1 on a 0–100 scale, using one-to-one Hungarian matching within each element type and the settings in [config.yaml](config.yaml). A type absent from both prediction and annotation scores 1; an unmatched type scores 0. Raw completion text is retained so parsing and metrics can be inspected independently.

## Single image

```bash
python -m inference.predict --variant elerpo --image example.png \
  --greedy --output outputs/example.jsonl
```

## Local models and data

```bash
python -m inference.predict \
  --model-path ./models/elerpo \
  --annotations ./my_data/annotations/test.jsonl \
  --data-root ./my_data \
  --output outputs/local.jsonl
```

Annotation records require `id`, `image`, `width`, `height`, `boxes`, and `types`. Image paths are relative to `--data-root`; boxes are pixel-space `[x1,y1,x2,y2]`; types are `text` or `visual`. Predictions are JSONL records containing `id` and `completion`.

## Options

- `--model-revision` and `--data-revision` pin Hugging Face revisions.
- `--batch-size` controls inference batches; the Transformers default is 1.
- `--limit 8` processes a prefix for a smoke test. Evaluate that output only against the same annotation prefix.
- `--greedy` selects deterministic decoding for applications. The default experiment settings use sampling with a fixed seed. Sampling results can vary with backend, batching, and library versions.
- `--device cpu` is supported by the Transformers backend but is slow.
- Install a compatible `vllm` separately to use `--backend vllm --batch-size 16`. Select a GPU with `CUDA_VISIBLE_DEVICES` when needed.

The image pipeline preserves the experiment's maximum image size, JPEG serialization, normalized coordinates, and ordered prompt variants. Detailed training settings are given in the paper.
