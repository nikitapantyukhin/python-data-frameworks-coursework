"""Общие функции для локальной выборки ручного обогащения HaGRID."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "hagrid_manual_enrichment.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_CLASSES = ("like", "dislike", "palm", "fist")
MANIFEST_COLUMNS = (
    "task_id",
    "image_name",
    "source_id",
    "canonical_class",
    "source_relative_path",
    "sha256",
    "dhash_64",
    "width",
    "height",
    "exif_orientation",
)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    source = config.get("source", {})
    if source.get("dataset_url") != "https://huggingface.co/datasets/CNGR/CN_Gesture_Recognition":
        raise ValueError("unexpected source dataset URL")
    if str(source.get("license", "")).lower() != "mit":
        raise ValueError("source license must be MIT")
    mapping = config.get("class_mapping")
    expected_mapping = {
        "thumbs_up": "like",
        "thumbs_down": "dislike",
        "palm": "palm",
        "fist_up": "fist",
    }
    if mapping != expected_mapping:
        raise ValueError("class_mapping must match the four source class directories")
    return config


def load_selection_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("selection manifest is empty")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dhash(path: Path, hash_size: int = 8) -> int:
    with Image.open(path) as image:
        grayscale = ImageOps.exif_transpose(image).convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
        pixels = list(grayscale.getdata())
    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def build_blank_task(task_id: int, image_name: str, source_id: str) -> dict[str, Any]:
    return {
        "id": task_id,
        "data": {
            "image": f"/data/local-files/?d=images/{image_name}",
            "source_id": source_id,
        },
    }


def label_studio_config() -> str:
    return """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="gesture" toName="image">
    <Label value="like" background="#2E7D32"/>
    <Label value="dislike" background="#C62828"/>
    <Label value="palm" background="#1565C0"/>
    <Label value="fist" background="#6A1B9A"/>
  </RectangleLabels>
</View>
"""
