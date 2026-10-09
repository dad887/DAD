from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".partial")
    staging.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")
    staging.replace(path)


def append_jsonl(file: Any, value: Any) -> None:
    file.write(json.dumps(jsonable(value), ensure_ascii=False) + "\n")
    file.flush()


def error_text(exc: Exception, api: Any) -> str:
    key = getattr(api, "key", None)
    return str(exc).replace(key, "[REDACTED]") if key else str(exc)


def slug(text: str, limit: int = 90) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:limit] or "item"


def parse_size(text: str) -> tuple[int, int]:
    width, height = map(int, text.lower().split("x"))
    if width <= 0 or height <= 0:
        raise ValueError("Canvas dimensions must be positive")
    return width, height


def prompt_text(path: Path, **values: Any) -> str:
    return path.read_text(encoding="utf-8").strip().format(**values)
