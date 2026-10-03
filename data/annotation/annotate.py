"""Annotate images with the Appendix B prompt using the Gemini API."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

from PIL import Image


def convert_response(text, width, height):
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("layers"), list):
        raise ValueError("Expected a JSON object containing a layers list")
    boxes = []
    types = []
    for layer in data["layers"]:
        box = layer.get("bbox")
        kind = layer.get("type")
        if kind not in ("t", "v") or not isinstance(box, list) or len(box) != 4:
            raise ValueError("Invalid layer schema")
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            for x in box
        ):
            raise ValueError("Bounding box coordinates must be finite numbers")
        if not all(0 <= x <= 1 for x in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Expected positive xyxy boxes normalized to [0,1]")
        pixels = [
            round(box[0] * width),
            round(box[1] * height),
            round(box[2] * width),
            round(box[3] * height),
        ]
        if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
            continue
        boxes.append(pixels)
        types.append("text" if kind == "t" else "visual")
    return boxes, types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model", required=True, help="Gemini model ID available to your API account"
    )
    p.add_argument(
        "--prompt", type=Path, default=Path(__file__).with_name("prompt.txt")
    )
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        p.error("Set GEMINI_API_KEY in your environment")
    if args.retries < 0:
        p.error("--retries must be non-negative")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    prompt = args.prompt.read_text()
    files = sorted(
        x
        for x in args.input_dir.rglob("*")
        if x.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and x.is_file()
    )
    if not files:
        p.error("No supported images found")
    done = set()
    if args.resume and args.output.exists():
        with args.output.open() as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.resume else "w") as out:
        for path in files:
            relative = path.relative_to(args.input_dir).as_posix()
            if relative in done:
                continue
            with Image.open(path) as im:
                image = im.convert("RGB")
            for attempt in range(args.retries + 1):
                try:
                    response = client.models.generate_content(
                        model=args.model,
                        contents=[prompt, image],
                        config=types.GenerateContentConfig(
                            temperature=0, response_mime_type="application/json"
                        ),
                    )
                    boxes, labels = convert_response(response.text, *image.size)
                    break
                except Exception:
                    if attempt == args.retries:
                        raise RuntimeError(
                            "Annotation failed for "
                            + relative
                            + "; rerun with --resume"
                        ) from None
                    time.sleep(2**attempt)
            record = {
                "id": relative,
                "image": relative,
                "width": image.width,
                "height": image.height,
                "boxes": boxes,
                "types": labels,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print("Annotated " + relative, flush=True)


if __name__ == "__main__":
    main()
