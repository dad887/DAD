# Detect Anything in Graphic Design

Code and public-data models for **Detect Anything in Graphic Design: Element-Level Rewards for Autoregressive Detection**.

- [Public dataset](https://huggingface.co/datasets/dad887/DAD)
- [Public experiment data](https://huggingface.co/datasets/dad887/DAD-Public-Repro): fixed SFT, RL, and test splits
- [Public experiment models](https://huggingface.co/dad887/DAD-Public-Models): SFT, GRPO, and EleRPO in one repository

The released models are the **public-data experiment models** based on Qwen3-VL-2B-Instruct. They are separate from the DAD model used for the paper's main experiments.

## Install

Use Python 3.10 or newer and a PyTorch installation suitable for your device.

```bash
pip install -r requirements.txt
```

## Detect and evaluate

Run from this repository's root. The command downloads the selected model and the test split, verifies image checksums, and writes one prediction per image.

```bash
python -m inference.predict --variant elerpo --output outputs/elerpo.jsonl
python -m inference.evaluate \
  --annotations public_data/annotations/test.jsonl \
  --predictions outputs/elerpo.jsonl
```

Choose `--variant sft`, `grpo`, or `elerpo`. For a quick check, add `--limit 8`; the full test split contains 5,000 examples. See [inference instructions](inference/README.md) for local files, single-image detection, decoding, and optional accelerated inference.

## Contents

| Directory | Contents |
|---|---|
| [elerpo](elerpo/README.md) | Element-level reward and advantage implementation, with integration examples |
| [inference](inference/README.md) | Model loading, prediction, data download, and evaluation |
| [data/annotation](data/annotation/README.md) | Appendix B prompt and model-calling script |
| [data/filtering](data/filtering/README.md) | Small-element selection and over-splitting filter |
| `data/element_replacement` | Reserved directory for the element-replacement pipeline |

Training settings are described in the paper. This release provides model weights and standalone inference/evaluation tools; it does not include training launchers.

## Output

Models generate layers in back-to-front order:

```text
x1,y1,x2,y2,t;x1,y1,x2,y2,v;
```

Coordinates use the normalized range `[0, 1000]`. `t` denotes text and `v` denotes a visual element. Dataset annotations use pixel-space `xyxy` boxes, with the image dimensions stored in each record.

## License

Code: [Apache 2.0](LICENSE). Model weights and datasets have their own license files and upstream notices in the corresponding Hugging Face repositories.
