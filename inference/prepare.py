"""Fetch only the requested public split and verify its image archives."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from huggingface_hub import hf_hub_download
from .common import read_jsonl, sha256


def prepare_dataset(root, repo, split="test", revision="main"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        return Path(
            hf_hub_download(
                repo_id=repo, repo_type="dataset", filename=name, revision=revision
            )
        )

    annotation_source = fetch(f"annotations/{split}.jsonl")
    annotation_target = root / "annotations" / f"{split}.jsonl"
    annotation_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(annotation_source, annotation_target)
    rows = read_jsonl(annotation_target)
    needed = {r["image"] for r in rows}
    image_manifest = read_jsonl(fetch("manifests/images.jsonl"))
    expected = {r["image"]: r for r in image_manifest if r["image"] in needed}
    if set(expected) != needed:
        raise ValueError("The image manifest does not cover this split")
    shards = json.loads(fetch("manifests/shards.json").read_text())
    for shard in shards:
        if shard["split"] != split:
            continue
        members = {k: v for k, v in expected.items() if v["shard"] == shard["file"]}
        if all(
            (root / k).is_file() and sha256(root / k) == v["sha256"]
            for k, v in members.items()
        ):
            continue
        archive = fetch(shard["file"])
        if sha256(archive) != shard["sha256"]:
            raise ValueError("Archive checksum mismatch: " + shard["file"])
        with tarfile.open(archive, "r:") as tf:
            for item in tf:
                path = PurePosixPath(item.name)
                if (
                    not item.isfile()
                    or path.is_absolute()
                    or ".." in path.parts
                    or item.name not in members
                ):
                    raise ValueError("Unexpected archive member: " + item.name)
                target = root.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_suffix(target.suffix + ".partial")
                with tf.extractfile(item) as src, temp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                if sha256(temp) != members[item.name]["sha256"]:
                    temp.unlink()
                    raise ValueError("Image checksum mismatch: " + item.name)
                temp.replace(target)
    for name, item in expected.items():
        if not (root / name).is_file() or sha256(root / name) != item["sha256"]:
            raise ValueError("Missing or corrupt image: " + name)
    return annotation_target
