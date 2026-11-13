"""Валидация подмножества HaGRID без изменения исходных данных.

Скрипт запускается из корня проекта командой
``python -m scripts.hagrid.validate_hagrid``. По умолчанию Pillow проверяет структуру
файлов с помощью ``Image.verify()``. Параметр ``--decode-images`` дополнительно
выполняет полное декодирование пикселей методом ``Image.load()``. Ни один из
режимов не изменяет изображения или аннотации.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import filecmp
import json
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "data" / "raw" / "hagrid"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ANNOTATION_FILES = (
    "annotations.json",
    "minimal-annotations.json",
    "exported_images.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Каталог HaGRID (по умолчанию: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--decode-images",
        action="store_true",
        help="Полностью декодировать пиксели средствами Pillow.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"top-level JSON value must be an object, got {type(value).__name__}")
    return value


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    print("HaGRID validation")
    print(f"Data root: {data_root}")
    print(f"Image check: {'full pixel decode' if args.decode_images else 'Image.verify structure check'}")
    if not data_root.is_dir():
        print(f"[ERROR] Data directory does not exist: {data_root}")
        print("RESULT: FAILED (1 critical error, 0 warnings)")
        return 1

    missing_json = [name for name in ANNOTATION_FILES if not (data_root / name).is_file()]
    if missing_json:
        print(f"[ERROR] Missing JSON files: {', '.join(missing_json)}")
        print("RESULT: FAILED (1 critical error, 0 warnings)")
        return 1

    loaded: dict[str, dict[str, Any]] = {}
    for name in ANNOTATION_FILES:
        try:
            loaded[name] = load_json(data_root / name)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"Cannot read {name}: {type(error).__name__}: {error}")
    if errors:
        for message in errors:
            print(f"[ERROR] {message}")
        print(f"RESULT: FAILED ({len(errors)} critical errors, 0 warnings)")
        return 1

    annotations = loaded["annotations.json"]
    minimal_annotations = loaded["minimal-annotations.json"]
    exported_images = loaded["exported_images.json"]

    image_paths = sorted(
        path
        for path in data_root.glob("*/*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        print(f"[ERROR] No images found one directory below {data_root}")
        print("RESULT: FAILED (1 critical error, 0 warnings)")
        return 1

    class_counts = Counter(path.parent.name for path in image_paths)
    extension_counts = Counter(path.suffix.lower() for path in image_paths)
    image_bytes = sum(path.stat().st_size for path in image_paths)
    paths_by_id: dict[str, list[Path]] = defaultdict(list)
    for path in image_paths:
        paths_by_id[path.stem].append(path)

    formats: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    dimensions: Counter[tuple[int, int]] = Counter()
    unreadable_images: list[str] = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                formats[image.format or "unknown"] += 1
                modes[image.mode] += 1
                dimensions[image.size] += 1
                if args.decode_images:
                    image.load()
                else:
                    image.verify()
        except (OSError, SyntaxError, ValueError) as error:
            unreadable_images.append(f"{path.relative_to(data_root)}: {error}")
    if unreadable_images:
        errors.append(f"Unreadable images: {len(unreadable_images):,}")

    full_pairs: set[tuple[str, str]] = set()
    minimal_pairs: set[tuple[str, str]] = set()
    exported_pairs: set[tuple[str, str]] = set()
    annotation_label_counts: Counter[str] = Counter()
    annotation_box_count = 0
    unique_users: set[str] = set()
    issue_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    warning_examples: dict[str, list[str]] = defaultdict(list)

    for class_name, records in annotations.items():
        if not isinstance(records, dict):
            issue_counts["full_class_value_not_object"] += 1
            continue
        for image_id, record in records.items():
            full_pairs.add((class_name, image_id))
            if not isinstance(record, dict):
                issue_counts["full_record_not_object"] += 1
                continue
            labels = record.get("labels")
            bboxes = record.get("bboxes")
            if not isinstance(labels, list) or not isinstance(bboxes, list):
                issue_counts["full_labels_or_bboxes_not_list"] += 1
                continue
            if len(labels) != len(bboxes):
                issue_counts["labels_bboxes_length_mismatch"] += 1
            annotation_label_counts.update(str(label) for label in labels)
            annotation_box_count += len(bboxes)
            user_id = record.get("user_id")
            if isinstance(user_id, str) and user_id:
                unique_users.add(user_id)
            else:
                warning_counts["missing_user_id"] += 1
                warning_examples["missing_user_id"].append(f"{class_name}/{image_id}")
            for bbox in bboxes:
                if not isinstance(bbox, list) or len(bbox) != 4:
                    issue_counts["invalid_bbox_shape"] += 1
                    continue
                if not all(isinstance(value, (int, float)) for value in bbox):
                    issue_counts["non_numeric_bbox"] += 1
                    continue
                x, y, width, height = bbox
                if x < 0 or y < 0 or x > 1 or y > 1 or width > 1 or height > 1:
                    issue_counts["bbox_component_outside_0_1"] += 1
                if width <= 0 or height <= 0:
                    issue_counts["non_positive_bbox_size"] += 1
                if x + width > 1.000001 or y + height > 1.000001:
                    warning_counts["bbox_exceeds_image"] += 1
                    warning_examples["bbox_exceeds_image"].append(
                        f"{class_name}/{image_id}: {bbox}"
                    )

    for class_name, records in minimal_annotations.items():
        if not isinstance(records, dict):
            issue_counts["minimal_class_value_not_object"] += 1
            continue
        for image_id, record in records.items():
            minimal_pairs.add((class_name, image_id))
            if not isinstance(record, dict):
                issue_counts["minimal_record_not_object"] += 1
                continue
            if not isinstance(record.get("labels"), list):
                issue_counts["minimal_labels_not_list"] += 1
            if not isinstance(record.get("bboxes"), list):
                issue_counts["minimal_bboxes_not_list"] += 1
            if "meta" not in record:
                issue_counts["minimal_meta_missing"] += 1

    for class_name, image_ids in exported_images.items():
        if not isinstance(image_ids, list):
            issue_counts["export_class_value_not_list"] += 1
            continue
        if len(image_ids) != len(set(image_ids)):
            issue_counts["duplicate_id_in_export_class"] += 1
        for image_id in image_ids:
            if not isinstance(image_id, str):
                issue_counts["non_string_export_id"] += 1
                continue
            exported_pairs.add((class_name, image_id))

    if full_pairs != minimal_pairs:
        issue_counts["full_minimal_key_mismatch"] += len(full_pairs ^ minimal_pairs)
    for class_name, image_id in full_pairs & minimal_pairs:
        full_record = annotations[class_name][image_id]
        minimal_record = minimal_annotations[class_name][image_id]
        if not isinstance(full_record, dict) or not isinstance(minimal_record, dict):
            continue
        for field in ("labels", "bboxes", "meta"):
            if minimal_record.get(field) != full_record.get(field):
                issue_counts[f"full_minimal_{field}_mismatch"] += 1

    file_pairs = {(path.parent.name, path.stem) for path in image_paths}
    duplicate_ids = {image_id: paths for image_id, paths in paths_by_id.items() if len(paths) > 1}
    if duplicate_ids:
        errors.append(f"Image identifiers repeated across class folders: {len(duplicate_ids):,}")
    checkpoint_paths = [path for path in image_paths if path.stem.endswith("-checkpoint")]
    checkpoint_copies = 0
    for path in checkpoint_paths:
        original = path.with_name(f"{path.stem.removesuffix('-checkpoint')}{path.suffix}")
        if original.is_file() and filecmp.cmp(path, original, shallow=False):
            checkpoint_copies += 1

    missing_full = file_pairs - full_pairs
    missing_minimal = file_pairs - minimal_pairs
    if missing_full:
        warnings.append(f"Images without full annotations: {len(missing_full):,}")
    if missing_minimal:
        warnings.append(f"Images without minimal annotations: {len(missing_minimal):,}")
    if full_pairs - file_pairs:
        errors.append(f"Full annotations without images: {len(full_pairs - file_pairs):,}")
    if minimal_pairs - file_pairs:
        errors.append(f"Minimal annotations without images: {len(minimal_pairs - file_pairs):,}")
    if file_pairs != exported_pairs:
        errors.append(
            "Export manifest mismatch: "
            f"missing={len(file_pairs - exported_pairs):,}, "
            f"extra={len(exported_pairs - file_pairs):,}"
        )
    if set(class_counts) != set(exported_images):
        errors.append("Class directories and export-manifest classes differ")
    if issue_counts:
        errors.append(f"Annotation/schema issues: {dict(sorted(issue_counts.items()))}")
    for warning_name, count in sorted(warning_counts.items()):
        warnings.append(f"{warning_name}: {count:,}")

    print(
        f"[INFO] Images: {len(image_paths):,}; classes: {len(class_counts)}; "
        f"bytes: {image_bytes:,}"
    )
    print(f"[INFO] Images per class: {dict(sorted(class_counts.items()))}")
    print(f"[INFO] Extensions: {dict(sorted(extension_counts.items()))}")
    print(f"[INFO] Pillow formats: {dict(sorted(formats.items()))}; modes: {dict(sorted(modes.items()))}")
    print(f"[INFO] Distinct dimensions: {len(dimensions)}; most common: {dimensions.most_common(10)}")
    print(
        f"[INFO] Annotation records: full={len(full_pairs):,}, "
        f"minimal={len(minimal_pairs):,}, export={len(exported_pairs):,}"
    )
    print(f"[INFO] Bounding boxes: {annotation_box_count:,}; labels: {dict(sorted(annotation_label_counts.items()))}")
    print(f"[INFO] Unique non-empty user_id: {len(unique_users):,}")
    print(
        f"[INFO] '-checkpoint' images: {len(checkpoint_paths)}; "
        f"byte-identical paired copies: {checkpoint_copies}"
    )
    for issue in unreadable_images[:10]:
        print(f"[ERROR] {issue}")
    for warning_name in sorted(warning_examples):
        for example in warning_examples[warning_name][:5]:
            print(f"[WARNING] Example {warning_name}: {example}")
    for class_name, image_id in sorted(missing_full)[:20]:
        print(f"[WARNING] No annotation: {class_name}/{image_id}")
    for message in errors:
        print(f"[ERROR] {message}")
    for message in warnings:
        print(f"[WARNING] {message}")
    status = "FAILED" if errors else "PASSED"
    print(f"RESULT: {status} ({len(errors)} critical errors, {len(warnings)} warnings)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
