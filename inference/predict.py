"""Run a public DAD model on a dataset split or individual image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import CONFIG_PATH, load_config, prepare_image, read_jsonl


def resolve_model(path, repo, variant, revision):
    if path:
        return str(path)
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=repo, revision=revision, allow_patterns=[variant + "/*"]
    )
    return str(Path(root) / variant)


def generate_transformers(model_path, rows, root, config, batch_size, device):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed

    set_seed(config["seed"])
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype=dtype)
        .to(device)
        .eval()
    )
    generation = {
        "max_new_tokens": config["max_new_tokens"],
        "do_sample": config["temperature"] > 0,
    }
    if generation["do_sample"]:
        generation.update(
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
        )
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        images = [
            prepare_image(
                root / r["image"], config["max_image_size"], config["jpeg_quality"]
            )
            for r in batch
        ]
        prompts = []
        for i, row in enumerate(batch, offset):
            prompt = config["prompts"][i % len(config["prompts"])]
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            ]
            prompts.append(
                processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        inputs = processor(
            text=prompts, images=images, padding=True, return_tensors="pt"
        ).to(device)
        inputs.pop("token_type_ids", None)
        # Variable image shapes can exceed the cuDNN SDPA plan support.
        with torch.inference_mode(), sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            generated = model.generate(**inputs, **generation)
        responses = processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for row, response in zip(batch, responses):
            yield {"id": row["id"], "completion": response}


def generate_vllm(model_path, rows, root, config, batch_size, device):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        seed=config["seed"],
        max_model_len=16384,
        gpu_memory_utilization=0.65,
        limit_mm_per_prompt={"image": 1},
    )
    params = SamplingParams(
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=config["top_k"],
        max_tokens=config["max_new_tokens"],
        seed=config["seed"],
    )
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        requests = []
        for i, row in enumerate(batch, offset):
            image = prepare_image(
                root / row["image"], config["max_image_size"], config["jpeg_quality"]
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": config["prompts"][i % len(config["prompts"])],
                        },
                    ],
                }
            ]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            requests.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        responses = llm.generate(requests, params, use_tqdm=False)
        for row, response in zip(batch, responses):
            yield {"id": row["id"], "completion": response.outputs[0].text}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["sft", "grpo", "elerpo"], default="elerpo")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--model-repo")
    p.add_argument("--model-revision", default="main")
    p.add_argument("--dataset-repo")
    p.add_argument("--data-revision", default="main")
    p.add_argument("--data-root", type=Path, default=Path("public_data"))
    p.add_argument("--split", choices=["sft", "rl", "test"], default="test")
    p.add_argument(
        "--annotations",
        type=Path,
        help="Use a local annotation JSONL instead of downloading a split",
    )
    p.add_argument("--image", type=Path, help="Run detection on one image")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument(
        "--backend", choices=["transformers", "vllm"], default="transformers"
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--greedy", action="store_true", help="Use deterministic decoding")
    p.add_argument("--limit", type=int, help="Process a prefix for a smoke test")
    args = p.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        p.error("Batch size and limit must be positive")
    if args.image and args.annotations:
        p.error("--image and --annotations are mutually exclusive")
    config = load_config(args.config)
    if args.greedy:
        config["temperature"] = 0.0
    if args.image:
        rows = [{"id": args.image.stem, "image": str(args.image.resolve())}]
    else:
        annotations = args.annotations
        if annotations is None:
            from .prepare import prepare_dataset

            annotations = prepare_dataset(
                args.data_root,
                args.dataset_repo or config["dataset_repo"],
                args.split,
                args.data_revision,
            )
        rows = read_jsonl(annotations)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        p.error("No input records")
    model_path = resolve_model(
        args.model_path,
        args.model_repo or config["model_repo"],
        args.variant,
        args.model_revision,
    )
    generate = (
        generate_transformers if args.backend == "transformers" else generate_vllm
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".partial")
    count = 0
    with temp.open("w") as f:
        for row in generate(
            model_path, rows, args.data_root, config, args.batch_size, args.device
        ):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            count += 1
            if count % 100 == 0:
                print(f"Processed {count}/{len(rows)}", flush=True)
    if count != len(rows):
        raise RuntimeError(f"Incomplete inference: {count}/{len(rows)}")
    temp.replace(args.output)
    print(f"Saved {count} predictions to {args.output}")


if __name__ == "__main__":
    main()
