"""Проверить локальный ручной отбор HaGRID без изменения данных."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from scripts.hagrid.manual_enrichment import (
    DEFAULT_CONFIG,
    IMAGE_SUFFIXES,
    TARGET_CLASSES,
    hamming_distance,
    load_config,
    load_selection_manifest,
    sha256_file,
)
from scripts.hagrid.prepare_manual_enrichment import validate_row


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "hagrid_manual_enrichment"
DEFAULT_HAGRID = ROOT / "data" / "raw" / "hagrid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hagrid-root", type=Path, default=DEFAULT_HAGRID)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def hagrid_hashes(root: Path, cache_path: Path) -> set[str]:
    paths = sorted(
        path
        for path in root.glob("*/*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if cache_path.is_file():
        cached = read_csv(cache_path)
        metadata = {
            row["relative_path"]: (int(row["size_bytes"]), int(row["mtime_ns"]))
            for row in cached
        }
        current = {
            path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in paths
        }
        if metadata == current:
            return {row["sha256"] for row in cached}
    return {sha256_file(path) for path in paths}


def main() -> int:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    hagrid_root = args.hagrid_root.expanduser().resolve()
    config = load_config(args.config.expanduser().resolve())
    manifests = root / "manifests"
    manifest = manifests / config["selection"]["manifest_name"]
    required = (
        manifest,
        manifests / "label_studio_tasks_blank.json",
        manifests / "label_studio_config.xml",
        manifests / "annotation_tracking_template.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(json.dumps({"status": "FAILED", "missing": missing}, ensure_ascii=False, indent=2))
        return 1

    errors: list[str] = []
    rows = load_selection_manifest(manifest)
    quota = int(config["selection"]["images_per_class"])
    expected_total = int(config["selection"]["expected_total"])
    counts = Counter(row["canonical_class"] for row in rows)
    if len(rows) != expected_total:
        errors.append(f"selection rows {len(rows)} != {expected_total}")
    if counts != Counter({name: quota for name in TARGET_CLASSES}):
        errors.append(f"class quotas differ: {dict(counts)}")
    for field in ("task_id", "image_name", "source_id", "sha256"):
        if len({row[field] for row in rows}) != len(rows):
            errors.append(f"duplicate {field} in selection manifest")

    selected_hashes: set[str] = set()
    for row in rows:
        image_path = root / "images" / row["image_name"]
        if not image_path.is_file():
            errors.append(f"missing image {row['image_name']}")
            continue
        try:
            validate_row(row, image_path)
            selected_hashes.add(row["sha256"])
        except (OSError, SyntaxError, ValueError) as error:
            errors.append(f"invalid {row['image_name']}: {error}")

    exact_overlap = 0
    if not hagrid_root.is_dir():
        errors.append(f"HaGRID root not found: {hagrid_root}")
    else:
        exact_overlap = len(
            selected_hashes
            & hagrid_hashes(hagrid_root, manifests / "hagrid_image_fingerprints.csv")
        )
        if exact_overlap:
            errors.append(f"exact overlap with HaGRID: {exact_overlap}")

    threshold = int(config["selection"]["near_duplicate_hamming_threshold"])
    near_pairs = 0
    fingerprints = [(row["image_name"], int(row["dhash_64"], 16)) for row in rows]
    for left_index, (_, left_hash) in enumerate(fingerprints):
        for _, right_hash in fingerprints[left_index + 1 :]:
            near_pairs += int(hamming_distance(left_hash, right_hash) <= threshold)

    tasks = json.loads((manifests / "label_studio_tasks_blank.json").read_text(encoding="utf-8"))
    if len(tasks) != expected_total:
        errors.append(f"blank task count {len(tasks)} != {expected_total}")
    expected_tasks = {
        (int(row["task_id"]), f"/data/local-files/?d=images/{row['image_name']}", row["source_id"])
        for row in rows
    }
    actual_tasks = {
        (task.get("id"), task.get("data", {}).get("image"), task.get("data", {}).get("source_id"))
        for task in tasks
    }
    if actual_tasks != expected_tasks:
        errors.append("blank tasks differ from selection manifest")
    if any("annotations" in task or "predictions" in task for task in tasks):
        errors.append("blank tasks contain annotations or predictions")
    for task in tasks:
        data = task.get("data", {})
        if not re.search(r"[?&]d=images/image_\d{3}\.(?:jpg|jpeg|png|webp)$", str(data.get("image", ""))):
            errors.append(f"non-neutral image URL in task {task.get('id')}")
        if not re.fullmatch(r"hagrid_manual:\d{3}", str(data.get("source_id", ""))):
            errors.append(f"non-neutral source_id in task {task.get('id')}")

    xml_root = ET.fromstring((manifests / "label_studio_config.xml").read_text(encoding="utf-8"))
    xml_labels = {element.attrib.get("value") for element in xml_root.iter("Label")}
    if xml_labels != set(TARGET_CLASSES):
        errors.append(f"Label Studio config classes differ: {xml_labels}")

    tracking = read_csv(manifests / "annotation_tracking_template.csv")
    if len(tracking) != expected_total * 2:
        errors.append(f"tracking rows {len(tracking)} != {expected_total * 2}")
    if Counter(row["participant_id"] for row in tracking) != {
        "participant_1": expected_total,
        "participant_2": expected_total,
    }:
        errors.append("tracking participant quotas differ")

    result = {
        "status": "FAILED" if errors else "PASSED",
        "errors": errors,
        "source": config["source"]["dataset_url"],
        "license": config["source"]["license"],
        "selected_images": len(rows),
        "selected_by_class": dict(sorted(counts.items())),
        "selected_unique_sha256": len(selected_hashes),
        "exact_overlap_with_hagrid": exact_overlap,
        "near_duplicate_pairs_at_threshold": near_pairs,
        "blank_tasks": len(tasks),
        "tracking_rows": len(tracking),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
