# Reference-driven layered design synthesis

Given reference images and their annotations, this pipeline generates layered graphic designs with element occlusions. It exports composited images, individual RGBA layers, amodal bounding boxes, and visible bounding boxes for graphic-design detection and image-to-layer decomposition.

## Pipeline

1. **Design planning:** a multimodal model reads each reference image and its annotations and produces three new background design specifications.
2. **Background generation and decomposition:** an image model generates each background and extracts its layers. The pipeline removes matte backgrounds, recomposes the layers, and reviews the reconstruction and background quality.
3. **Foreground generation:** the planner selects foreground objects, which are generated individually and converted into transparent assets.
4. **Layout and occlusion:** the pipeline searches object positions, scales, layer order, and decorative-layer placement to compose the final design.
5. **Export:** images, layers, and aligned annotations are saved. Recomposition from the saved layers is checked against the exported image.

## Installation

Use Python 3.10 or newer. Run the following commands from the DAD repository root:

```bash
cd data/element_replacement
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Model inference uses remote APIs; a local GPU is not required. Set `LLM_API_KEY` or `OPENAI_API_KEY` to a key with access to the selected text and image models:

```bash
export LLM_API_KEY="YOUR_API_KEY"
```

For an OpenAI-compatible service, set `LLM_BASE_URL` or pass `--base-url`. The service must support streaming chat completions with image inputs, image generation, and image editing. Image responses must include base64-encoded image data.

## Input

Create a JSON array of references, for example `references.json`:

```json
[
  {
    "reference_id": "education",
    "image": "images/reference_001.png",
    "annotations": {
      "caption": "A playful education poster with a compact headline and colorful decorative shapes.",
      "layer_caption": {}
    }
  }
]
```

Provide the referenced image file. Image paths are resolved relative to the input JSON file. `annotations` supplies design context to the planner; it can include a caption and descriptions of the reference layers.

## Run

Run from `data/element_replacement` with the environment above activated:

```bash
python run_pipeline.py \
  --references references.json \
  --output output/demo \
  --variants 1
```

Each reference produces three background candidates. Each accepted background produces up to `--variants` final designs; the default is 30. Repeating a run with the same inputs and settings reuses completed results.

| Option | Default | Purpose |
|---|---|---|
| `--planner-model` | `gpt-5.5` | Background specifications and foreground planning |
| `--layer-model` | `gpt-5-mini` | Layer planning and reconstruction review |
| `--review-model` | `gpt-5.5` | Background quality review |
| `--image-model` | `gpt-image-2` | Image generation and layer extraction |
| `--image-quality` | `low` | Image API quality setting |
| `--canvas-size` | `1024x1536` | Output canvas; also supports `1536x1024` and `1024x1024` |
| `--concurrency` | `10` | Maximum concurrent model requests |
| `--variants` | `30` | Final designs per accepted background |
| `--background-score-threshold` | `5` | Minimum background review score, from 1 to 5 |

Model names can be replaced with compatible models available through the configured service. Use `python run_pipeline.py --help` for all options.

## Output

```text
output/demo/
├── annotations.jsonl         # One record per final design
├── images/                   # Composited PNG images
├── layers/                   # Full-canvas RGBA layers for each design
├── specifications.json       # Background specifications and generation prompts
├── background_results.jsonl  # Background outcomes and failure details
├── reference_failures.json   # Reference-planning failures
├── summary.json              # Counts and elapsed time
├── api_usage.json            # Recorded model usage
├── input_snapshot.json       # Input and configuration snapshot for resuming
└── work/                     # Intermediate images, plans, and cached annotations
```

In each annotation record:

- `image` is the composited image path relative to the output directory.
- `boxes` and `visible_bboxes` contain amodal and visible boxes in pixel-space `[x0, y0, x1, y1]` coordinates.
- `types` contains `text` or `visual`, aligned with `boxes` and `target_layer_ids`.
- `layers` records the individual layer paths, layer order, and target metadata. Base layers and shadows are excluded from detection targets.
- `occlusion_stats` and `occlusion_events` describe element visibility and pairwise occlusion.
- `recompose_reproducibility` reports whether the saved layers exactly reproduce the final image.

## Offline tests

The included tests use simulated model responses and require no API key:

```bash
python -m unittest discover -s tests -v
```

They cover the complete pipeline, resuming, rejected backgrounds and layers, annotation alignment, recomposition, masking, and occlusion measurements.

## Code structure

| Path | Purpose |
|---|---|
| `run_pipeline.py` | Pipeline entry point |
| `config.py` | Configuration and command-line options |
| `api.py` | Model requests and concurrency control |
| `background/` | Design specifications, generation, decomposition, and review |
| `foreground/` | Foreground planning, generation, layout, occlusion, and export |
| `utils/` | Geometry, masking, image operations, and file I/O |
| `tests/` | Offline pipeline tests |

## License

This code is covered by the repository's [Apache 2.0 license](../../LICENSE).
