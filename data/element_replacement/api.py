from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from openai import NOT_GIVEN, OpenAI
from PIL import Image

from config import Config


class ModelClient:
    def __init__(self, config: Config):
        self.config = config
        self.key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise ValueError("Set LLM_API_KEY or OPENAI_API_KEY before running")
        self.client = OpenAI(api_key=self.key, base_url=config.base_url, timeout=config.request_timeout, max_retries=1)
        # 所有阶段共享同一个信号量，限制同时进行的模型请求数。
        self.slots = threading.BoundedSemaphore(config.concurrency)
        self.lock = threading.Lock()
        self.usage: list[dict[str, Any]] = []

    def _record(self, model: str, operation: str, usage: Any, elapsed: float) -> None:
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        with self.lock:
            self.usage.append(
                {"model": model, "operation": operation, "usage": usage or {}, "elapsed_sec": round(elapsed, 3)}
            )

    def json(self, prompt: str, images: list[Path], model: str) -> Any:
        content = []
        for path in images:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((1024, 1024))
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=90)
            encoded = base64.b64encode(buffer.getvalue()).decode()
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}})
        content.append({"type": "text", "text": prompt})
        with self.slots:
            started = time.monotonic()
            stream = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=8000,
                reasoning_effort="low" if model.startswith("gpt-5") else NOT_GIVEN,
                stream=True,
                stream_options={"include_usage": True},
            )
            parts, usage, returned_model = [], None, model
            for chunk in stream:
                usage = chunk.usage or usage
                returned_model = chunk.model or returned_model
                parts.extend(choice.delta.content for choice in chunk.choices if choice.delta.content)
            self._record(returned_model, "chat.completions", usage, time.monotonic() - started)
        # 去掉模型可能包裹的 Markdown 代码块标记。
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", "".join(parts).strip())
        return json.loads(text)

    def _image(self, prompt: str, path: Path, size: str, source: Path | None = None) -> Path:
        # 图片已存在时直接复用，中断后重跑不会重复请求。
        if path.exists():
            with Image.open(path) as image:
                image.load()
                if image.size != tuple(map(int, size.split("x"))):
                    raise ValueError(f"Cached image size differs from requested size: {path}")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            "model": self.config.image_model,
            "quality": self.config.image_quality,
            "size": size,
            "n": 1,
            "prompt": prompt,
        }
        with self.slots:
            started = time.monotonic()
            if source is None:
                response = self.client.images.generate(**params)
                operation = "images.generate"
            else:
                with source.open("rb") as image:
                    response = self.client.images.edit(image=image, **params)
                operation = "images.edit"
            self._record(
                self.config.image_model, operation, getattr(response, "usage", None), time.monotonic() - started
            )
        data = base64.b64decode(response.data[0].b64_json)
        with Image.open(BytesIO(data)) as image:
            if image.size != tuple(map(int, size.split("x"))):
                raise ValueError(f"Generated image size differs from requested size: {image.size}")
            image.save(path, format="PNG")
        return path

    def generate(self, prompt: str, path: Path, size: str) -> Path:
        return self._image(prompt, path, size)

    def extract(self, prompt: str, source: Path, path: Path, size: str) -> Path:
        return self._image(prompt, path, size, source)
