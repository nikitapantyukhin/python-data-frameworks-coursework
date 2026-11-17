"""Проверить ручной отбор 200 изображений и подготовить пустые задачи Label Studio."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
from typing import Any

from PIL import Image, ImageOps

from scripts.hagrid.manual_enrichment import (
    DEFAULT_CONFIG,
    TARGET_CLASSES,
    build_blank_task,
    image_dhash,
    label_studio_config,
    load_config,
    load_selection_manifest,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = ROOT / "data" / "raw" / "cn_gesture_recognition"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "hagrid_manual_enrichment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def write_if_changed(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == payload:
        return "unchanged"
    path.write_bytes(payload)
    return "written"


def copy_if_needed(source: Path, destination: Path, expected_sha256: str) -> str:
    if destination.is_file():
        if sha256_file(destination) != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for existing {destination}")
        return "unchanged"
    if not source.is_file():
        raise FileNotFoundError(
            f"Не найден {destination.name}; для восстановления нужен исходный файл {source}"
        )
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"SHA-256 source mismatch for {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return "copied"


def validate_row(row: dict[str, str], image_path: Path) -> None:
    if row["canonical_class"] not in TARGET_CLASSES:
        raise ValueError(f"unknown class for {row['image_name']}: {row['canonical_class']}")
    if sha256_file(image_path) != row["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {row['image_name']}")
    with Image.open(image_path) as image:
        image.verify()
    with Image.open(image_path) as image:
        size = ImageOps.exif_transpose(image).size
        orientation = int(image.getexif().get(274, 1))
    if size != (int(row["width"]), int(row["height"])):
        raise ValueError(f"dimension mismatch for {row['image_name']}")
    if orientation != int(row["exif_orientation"]):
        raise ValueError(f"EXIF orientation mismatch for {row['image_name']}")
    if f"{image_dhash(image_path):016x}" != row["dhash_64"].lower():
        raise ValueError(f"dHash mismatch for {row['image_name']}")


def tracking_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "participant_id": participant,
            "task_id": row["task_id"],
            "image_name": row["image_name"],
            "annotation_mode": "",
            "started_at": "",
            "finished_at": "",
            "duration_seconds": "",
            "bbox_review_status": "",
            "notes": "",
        }
        for participant in ("participant_1", "participant_2")
        for row in rows
    ]


def csv_payload(rows: list[dict[str, str]]) -> bytes:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config.expanduser().resolve())
    output = args.output_dir.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else output / "manifests" / config["selection"]["manifest_name"]
    )
    rows = load_selection_manifest(manifest)
    quota = int(config["selection"]["images_per_class"])
    expected_total = int(config["selection"]["expected_total"])
    counts = Counter(row["canonical_class"] for row in rows)
    if len(rows) != expected_total or counts != Counter({name: quota for name in TARGET_CLASSES}):
        raise ValueError(f"unexpected class quotas: rows={len(rows)}, classes={dict(counts)}")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate task_id in selection manifest")
    if len({row["image_name"] for row in rows}) != len(rows):
        raise ValueError("duplicate image_name in selection manifest")
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate source_id in selection manifest")
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("duplicate selected image content")

    copy_status: Counter[str] = Counter()
    for row in rows:
        destination = output / "images" / row["image_name"]
        relative = row["source_relative_path"].strip()
        if destination.is_file():
            copy_status[copy_if_needed(destination, destination, row["sha256"])] += 1
        elif relative:
            copy_status[copy_if_needed(source_root / relative, destination, row["sha256"])] += 1
        else:
            raise FileNotFoundError(
                f"Не найден {destination}; manifest не содержит source_relative_path для восстановления"
            )
        validate_row(row, destination)

    tasks = [
        build_blank_task(int(row["task_id"]), row["image_name"], row["source_id"])
        for row in rows
    ]
    manifests = output / "manifests"
    writes = {
        "label_studio_tasks_blank.json": write_if_changed(
            manifests / "label_studio_tasks_blank.json",
            (json.dumps(tasks, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        ),
        "label_studio_config.xml": write_if_changed(
            manifests / "label_studio_config.xml", label_studio_config().encode("utf-8")
        ),
    }
    tracking = manifests / "annotation_tracking_template.csv"
    if not tracking.exists():
        writes[tracking.name] = write_if_changed(tracking, csv_payload(tracking_rows(rows)))
    else:
        writes[tracking.name] = "preserved"

    result: dict[str, Any] = {
        "status": "PASSED",
        "source": config["source"]["dataset_url"],
        "license": config["source"]["license"],
        "selected_images": len(rows),
        "selected_by_class": dict(sorted(counts.items())),
        "image_files": dict(sorted(copy_status.items())),
        "writes": writes,
        "annotations_created": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
