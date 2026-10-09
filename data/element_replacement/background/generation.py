from pathlib import Path
from typing import Any


def generate_background(spec: dict[str, Any], api: Any, directory: Path) -> Path:
    return api.generate(spec["image_prompt"], directory / "background.png", spec["image_size"])
