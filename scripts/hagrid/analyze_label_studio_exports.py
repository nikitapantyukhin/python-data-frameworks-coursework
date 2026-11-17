"""Проверить и сравнить два JSON-экспорта Label Studio для 200 изображений."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any
from urllib.parse import parse_qs, urlsplit

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_ROOT = ROOT / "data" / "processed" / "hagrid_manual_enrichment" / "images"
DEFAULT_JSON_REPORT = (
    ROOT
    / "data"
    / "processed"
    / "hagrid_manual_enrichment"
    / "manifests"
    / "exports"
    / "quality_summary.json"
)
DEFAULT_MARKDOWN_REPORT = ROOT / "reports" / "hagrid_manual_annotation_quality.md"
ALLOWED_LABELS = ("like", "dislike", "palm", "fist")


@dataclass(frozen=True)
class Box:
    label: str
    x: float
    y: float
    width: float
    height: float
    original_width: int
    original_height: int
    rotation: float
    image_rotation: float

    @property
    def signature(self) -> tuple[Any, ...]:
        return (
            self.label,
            self.x,
            self.y,
            self.width,
            self.height,
            self.original_width,
            self.original_height,
            self.rotation,
            self.image_rotation,
        )


@dataclass
class SelectedTask:
    source_id: str
    image_url: str
    task_id: Any
    project: Any
    annotation_id: Any
    completed_by: Any
    annotation_created_at: Any
    annotation_updated_at: Any
    boxes: list[Box]
    active_annotations: int
    result_signature: str


@dataclass
class ParsedExport:
    path: Path
    tasks: dict[str, SelectedTask]
    summary: dict[str, Any]
    errors: list[str]
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-1", type=Path, required=True)
    parser.add_argument("--participant-2", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    parser.add_argument("--expected-tasks", type=int, default=200)
    parser.add_argument("--expected-per-class", type=int, default=50)
    parser.add_argument(
        "--bbox-tolerance-percent",
        type=float,
        default=1e-4,
        help="Допуск для округления координат Label Studio в процентах.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_result_signature(results: Any) -> str:
    """Сигнатура содержательной части result без служебных полей Label Studio."""

    if not isinstance(results, list):
        return "<invalid>"
    normalized = []
    for result in results:
        if not isinstance(result, dict):
            normalized.append(result)
            continue
        normalized.append(
            {
                "original_width": result.get("original_width"),
                "original_height": result.get("original_height"),
                "image_rotation": result.get("image_rotation", 0),
                "from_name": result.get("from_name"),
                "to_name": result.get("to_name"),
                "type": result.get("type"),
                "value": result.get("value"),
            }
        )
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def annotation_sort_key(annotation: dict[str, Any], position: int) -> tuple[str, str, int]:
    return (
        str(annotation.get("updated_at") or annotation.get("created_at") or ""),
        str(annotation.get("created_at") or ""),
        position,
    )


def image_name_from_url(image_url: str) -> str | None:
    parsed = urlsplit(image_url)
    values = parse_qs(parsed.query).get("d", [])
    if len(values) != 1:
        return None
    relative = Path(values[0])
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
        return None
    if relative.parts[0] != "images":
        return None
    return relative.name


def parse_box(
    result: Any,
    context: str,
    errors: list[str],
    warnings: list[str],
    tolerance: float,
) -> tuple[Box | None, bool]:
    if not isinstance(result, dict):
        errors.append(f"{context}: result is not an object")
        return None, False
    if result.get("type") != "rectanglelabels":
        errors.append(f"{context}: unsupported result type {result.get('type')!r}")
        return None, False
    value = result.get("value")
    if not isinstance(value, dict):
        errors.append(f"{context}: value is not an object")
        return None, False
    labels = value.get("rectanglelabels")
    if not isinstance(labels, list) or len(labels) != 1 or labels[0] not in ALLOWED_LABELS:
        errors.append(f"{context}: expected exactly one of {ALLOWED_LABELS}, got {labels!r}")
        return None, False
    numeric_fields = {
        "x": value.get("x"),
        "y": value.get("y"),
        "width": value.get("width"),
        "height": value.get("height"),
        "rotation": value.get("rotation", 0),
        "image_rotation": result.get("image_rotation", 0),
        "original_width": result.get("original_width"),
        "original_height": result.get("original_height"),
    }
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in numeric_fields.values()):
        errors.append(f"{context}: bbox contains missing or non-numeric fields")
        return None, False
    if any(not math.isfinite(float(item)) for item in numeric_fields.values()):
        errors.append(f"{context}: bbox contains a non-finite number")
        return None, False
    box = Box(label=labels[0], **{key: numeric_fields[key] for key in numeric_fields})
    if box.original_width <= 0 or box.original_height <= 0:
        errors.append(f"{context}: original image dimensions must be positive")
        return None, False
    if box.width <= 0 or box.height <= 0:
        errors.append(f"{context}: bbox dimensions must be positive")
        return None, False
    if box.rotation != 0 or box.image_rotation != 0:
        errors.append(f"{context}: rotated bbox is outside the current comparison protocol")
        return None, False
    edges = (box.x, box.y, box.x + box.width, box.y + box.height)
    if min(edges[:2]) < -tolerance or max(edges[2:]) > 100 + tolerance:
        errors.append(f"{context}: bbox exceeds [0, 100] beyond tolerance: {edges}")
        return None, False
    tolerated = min(edges[:2]) < 0 or max(edges[2:]) > 100
    if tolerated:
        warnings.append(f"{context}: bbox touches boundary within rounding tolerance: {edges}")
    return box, tolerated


def parse_export(
    path: Path,
    image_root: Path,
    expected_tasks: int,
    expected_per_class: int,
    tolerance: float,
    image_size_cache: dict[str, tuple[int, int]],
) -> ParsedExport:
    path = path.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return ParsedExport(path, {}, {}, [f"cannot read export: {error}"], [])
    if not isinstance(payload, list):
        return ParsedExport(path, {}, {}, ["top-level JSON value must be an array"], [])

    selected: dict[str, SelectedTask] = {}
    seen_images: Counter[str] = Counter()
    task_ids: Counter[str] = Counter()
    active_annotations_total = 0
    cancelled_annotations = 0
    empty_annotations = 0
    tasks_without_active = 0
    tasks_with_multiple = 0
    repeated_identical_submissions = 0
    repeated_divergent_variants = 0
    duplicate_boxes = 0
    tolerated_boxes = 0
    prediction_comparisons = 0
    changed_from_prediction = 0
    missing_images = 0
    dimension_mismatches = 0

    for task_position, task in enumerate(payload):
        context = f"task[{task_position}]"
        if not isinstance(task, dict):
            errors.append(f"{context}: task is not an object")
            continue
        data = task.get("data")
        if not isinstance(data, dict):
            errors.append(f"{context}: data is not an object")
            continue
        source_id = data.get("source_id")
        image_url = data.get("image")
        if not isinstance(source_id, str) or not source_id:
            errors.append(f"{context}: missing source_id")
            continue
        if not isinstance(image_url, str) or not image_url:
            errors.append(f"{context}: missing image URL")
            continue
        if source_id in selected:
            errors.append(f"{context}: duplicate source_id {source_id}")
            continue
        seen_images[image_url] += 1
        task_ids[str(task.get("id"))] += 1

        annotations = task.get("annotations", [])
        if not isinstance(annotations, list):
            errors.append(f"{context}: annotations is not an array")
            continue
        cancelled_annotations += sum(
            1 for annotation in annotations if isinstance(annotation, dict) and annotation.get("was_cancelled")
        )
        empty_annotations += sum(
            1
            for annotation in annotations
            if isinstance(annotation, dict) and not annotation.get("was_cancelled") and not annotation.get("result")
        )
        active = [
            (position, annotation)
            for position, annotation in enumerate(annotations)
            if isinstance(annotation, dict)
            and not annotation.get("was_cancelled")
            and isinstance(annotation.get("result"), list)
            and annotation["result"]
        ]
        active_annotations_total += len(active)
        if not active:
            tasks_without_active += 1
            errors.append(f"{context} {source_id}: no active non-empty annotation")
            continue
        if len(active) > 1:
            tasks_with_multiple += 1
            signatures = {canonical_result_signature(annotation["result"]) for _, annotation in active}
            repeated_identical_submissions += len(active) - len(signatures)
            repeated_divergent_variants += len(signatures) - 1
            if len(signatures) > 1:
                warnings.append(
                    f"{context} {source_id}: multiple active annotations differ; latest timestamp/order selected"
                )
        selected_position, annotation = max(
            active, key=lambda item: annotation_sort_key(item[1], item[0])
        )
        del selected_position

        boxes: list[Box] = []
        for result_position, result in enumerate(annotation["result"]):
            box, tolerated = parse_box(
                result,
                f"{context} {source_id} result[{result_position}]",
                errors,
                warnings,
                tolerance,
            )
            if box is not None:
                boxes.append(box)
                tolerated_boxes += int(tolerated)
        box_counts = Counter(box.signature for box in boxes)
        duplicate_boxes += sum(count - 1 for count in box_counts.values() if count > 1)

        prediction = annotation.get("prediction")
        if isinstance(prediction, dict) and isinstance(prediction.get("result"), list):
            prediction_comparisons += 1
            if canonical_result_signature(annotation["result"]) != canonical_result_signature(
                prediction["result"]
            ):
                changed_from_prediction += 1

        image_name = image_name_from_url(image_url)
        if image_name is None:
            errors.append(f"{context} {source_id}: unsupported local image URL {image_url!r}")
        else:
            image_path = image_root / image_name
            if image_name not in image_size_cache:
                try:
                    with Image.open(image_path) as image:
                        image_size_cache[image_name] = ImageOps.exif_transpose(image).size
                except (OSError, ValueError) as error:
                    errors.append(f"{context} {source_id}: cannot open {image_name}: {error}")
                    missing_images += 1
            actual_size = image_size_cache.get(image_name)
            if actual_size is not None:
                for box in boxes:
                    if (box.original_width, box.original_height) != actual_size:
                        dimension_mismatches += 1
                        errors.append(
                            f"{context} {source_id}: bbox dimensions "
                            f"{(box.original_width, box.original_height)} != image {actual_size}"
                        )

        selected[source_id] = SelectedTask(
            source_id=source_id,
            image_url=image_url,
            task_id=task.get("id"),
            project=task.get("project"),
            annotation_id=annotation.get("id"),
            completed_by=annotation.get("completed_by"),
            annotation_created_at=annotation.get("created_at"),
            annotation_updated_at=annotation.get("updated_at"),
            boxes=boxes,
            active_annotations=len(active),
            result_signature=canonical_result_signature(annotation["result"]),
        )

    duplicate_image_tasks = sum(count - 1 for count in seen_images.values() if count > 1)
    duplicate_task_ids = sum(count - 1 for count in task_ids.values() if count > 1)
    if len(payload) != expected_tasks:
        errors.append(f"task count {len(payload)} != expected {expected_tasks}")
    if duplicate_image_tasks:
        errors.append(f"duplicate image tasks: {duplicate_image_tasks}")
    if duplicate_task_ids:
        errors.append(f"duplicate task IDs: {duplicate_task_ids}")

    labels = Counter(box.label for task in selected.values() for box in task.boxes)
    selected_box_count = sum(labels.values())
    one_box_per_task = all(len(task.boxes) == 1 for task in selected.values())
    strict_class_quota = expected_tasks == expected_per_class * len(ALLOWED_LABELS)
    if one_box_per_task and len(selected) == expected_tasks and strict_class_quota:
        expected_distribution = {label: expected_per_class for label in ALLOWED_LABELS}
        if {label: labels[label] for label in ALLOWED_LABELS} != expected_distribution:
            errors.append(f"class distribution differs: {dict(labels)}")
    elif selected and not one_box_per_task:
        warnings.append("not every selected task has exactly one bbox; per-image class balance is not inferred")

    summary = {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "tasks": len(payload),
        "unique_source_ids": len(selected),
        "unique_images": len(seen_images),
        "active_annotations": active_annotations_total,
        "selected_annotations": len(selected),
        "tasks_without_active_annotations": tasks_without_active,
        "cancelled_annotations": cancelled_annotations,
        "empty_active_annotations": empty_annotations,
        "tasks_with_multiple_active_annotations": tasks_with_multiple,
        "repeated_identical_submissions": repeated_identical_submissions,
        "repeated_divergent_variants": repeated_divergent_variants,
        "duplicate_image_tasks": duplicate_image_tasks,
        "duplicate_task_ids": duplicate_task_ids,
        "duplicate_boxes_in_selected_annotations": duplicate_boxes,
        "selected_boxes": selected_box_count,
        "class_distribution": {label: labels[label] for label in ALLOWED_LABELS},
        "tolerated_boundary_roundings": tolerated_boxes,
        "missing_or_unreadable_images": missing_images,
        "image_dimension_mismatches": dimension_mismatches,
        "prediction_comparisons": prediction_comparisons,
        "tasks_changed_from_imported_prediction": changed_from_prediction,
        "project_values": sorted({str(task.project) for task in selected.values()}),
        "completed_by_values": sorted({str(task.completed_by) for task in selected.values()}),
    }
    return ParsedExport(path, selected, summary, errors, warnings)


def bbox_iou(left: Box, right: Box) -> float:
    if left.signature == right.signature:
        return 1.0
    intersection_width = max(0.0, min(left.x + left.width, right.x + right.width) - max(left.x, right.x))
    intersection_height = max(
        0.0, min(left.y + left.height, right.y + right.height) - max(left.y, right.y)
    )
    intersection = intersection_width * intersection_height
    union = left.width * left.height + right.width * right.height - intersection
    if union <= 0:
        return 0.0
    return min(1.0, max(0.0, intersection / union))


def best_box_matching(left: list[Box], right: list[Box]) -> tuple[list[tuple[int, int, float]], int, int]:
    """Максимизировать сумму IoU внутри одного класса без сторонних зависимостей."""

    swapped = False
    small, large = left, right
    if len(small) > len(large):
        small, large = large, small
        swapped = True

    @lru_cache(maxsize=None)
    def solve(index: int, used_mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if index == len(small):
            return 0.0, ()
        best_score = -1.0
        best_pairs: tuple[tuple[int, int], ...] = ()
        for large_index in range(len(large)):
            if used_mask & (1 << large_index):
                continue
            score, pairs = solve(index + 1, used_mask | (1 << large_index))
            score += bbox_iou(small[index], large[large_index])
            candidate = ((index, large_index),) + pairs
            if score > best_score or (math.isclose(score, best_score) and candidate < best_pairs):
                best_score = score
                best_pairs = candidate
        return best_score, best_pairs

    if not small:
        return [], len(left), len(right)
    _, index_pairs = solve(0, 0)
    pairs: list[tuple[int, int, float]] = []
    for small_index, large_index in index_pairs:
        left_index, right_index = (
            (large_index, small_index) if swapped else (small_index, large_index)
        )
        pairs.append((left_index, right_index, bbox_iou(left[left_index], right[right_index])))
    return pairs, len(left) - len(pairs), len(right) - len(pairs)


def cohens_kappa(left: list[str], right: list[str]) -> float | None:
    if not left or len(left) != len(right):
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        left_counts[label] / len(left) * right_counts[label] / len(right)
        for label in set(left_counts) | set(right_counts)
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1 - expected)


def compare_exports(left: ParsedExport, right: ParsedExport) -> dict[str, Any]:
    shared = sorted(set(left.tasks) & set(right.tasks))
    only_left = sorted(set(left.tasks) - set(right.tasks))
    only_right = sorted(set(right.tasks) - set(left.tasks))
    class_multiset_agreement = 0
    single_left: list[str] = []
    single_right: list[str] = []
    ious: list[float] = []
    exact_box_matches = 0
    unmatched_left = 0
    unmatched_right = 0
    differing_tasks: list[dict[str, Any]] = []
    metadata_overlap = Counter()

    for source_id in shared:
        left_task = left.tasks[source_id]
        right_task = right.tasks[source_id]
        if left_task.task_id == right_task.task_id:
            metadata_overlap["same_task_id"] += 1
        if left_task.annotation_id == right_task.annotation_id:
            metadata_overlap["same_annotation_id"] += 1
        if left_task.completed_by == right_task.completed_by:
            metadata_overlap["same_completed_by"] += 1
        if left_task.project == right_task.project:
            metadata_overlap["same_project"] += 1
        if left_task.annotation_created_at == right_task.annotation_created_at:
            metadata_overlap["same_annotation_created_at"] += 1
        if left_task.annotation_updated_at == right_task.annotation_updated_at:
            metadata_overlap["same_annotation_updated_at"] += 1

        left_classes = sorted(box.label for box in left_task.boxes)
        right_classes = sorted(box.label for box in right_task.boxes)
        class_equal = left_classes == right_classes
        class_multiset_agreement += int(class_equal)
        if len(left_task.boxes) == len(right_task.boxes) == 1:
            single_left.append(left_task.boxes[0].label)
            single_right.append(right_task.boxes[0].label)

        task_ious: list[float] = []
        task_unmatched_left = 0
        task_unmatched_right = 0
        for label in ALLOWED_LABELS:
            left_boxes = [box for box in left_task.boxes if box.label == label]
            right_boxes = [box for box in right_task.boxes if box.label == label]
            pairs, missing_left, missing_right = best_box_matching(left_boxes, right_boxes)
            values = [value for _, _, value in pairs]
            task_ious.extend(values)
            ious.extend(values)
            exact_box_matches += sum(math.isclose(value, 1.0, abs_tol=1e-12) for value in values)
            task_unmatched_left += missing_left
            task_unmatched_right += missing_right
        unmatched_left += task_unmatched_left
        unmatched_right += task_unmatched_right
        if (
            not class_equal
            or task_unmatched_left
            or task_unmatched_right
            or any(not math.isclose(value, 1.0, abs_tol=1e-12) for value in task_ious)
        ):
            differing_tasks.append(
                {
                    "source_id": source_id,
                    "image": left_task.image_url,
                    "participant_1_classes": left_classes,
                    "participant_2_classes": right_classes,
                    "matched_ious": task_ious,
                    "unmatched_participant_1": task_unmatched_left,
                    "unmatched_participant_2": task_unmatched_right,
                }
            )

    class_matches = sum(a == b for a, b in zip(single_left, single_right))
    comparison = {
        "shared_source_ids": len(shared),
        "only_participant_1": only_left,
        "only_participant_2": only_right,
        "same_image_urls": sum(
            left.tasks[source_id].image_url == right.tasks[source_id].image_url for source_id in shared
        ),
        "class_multiset_agreement_tasks": class_multiset_agreement,
        "single_label_tasks": len(single_left),
        "single_label_class_matches": class_matches,
        "percentage_agreement": class_matches / len(single_left) if single_left else None,
        "cohens_kappa": cohens_kappa(single_left, single_right),
        "matched_boxes": len(ious),
        "exact_box_matches": exact_box_matches,
        "unmatched_boxes_participant_1": unmatched_left,
        "unmatched_boxes_participant_2": unmatched_right,
        "iou_min": min(ious) if ious else None,
        "iou_mean": mean(ious) if ious else None,
        "iou_median": median(ious) if ious else None,
        "iou_at_least_0_5": sum(value >= 0.5 for value in ious),
        "iou_at_least_0_75": sum(value >= 0.75 for value in ious),
        "iou_at_least_0_9": sum(value >= 0.9 for value in ious),
        "differing_tasks": differing_tasks,
        "metadata_overlap": dict(metadata_overlap),
    }
    return comparison


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def format_number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_markdown(summary: dict[str, Any]) -> str:
    first, second = summary["exports"]
    comparison = summary["comparison"]
    rows = [
        f"| `{label}` | {first['class_distribution'][label]} | {second['class_distribution'][label]} |"
        for label in ALLOWED_LABELS
    ]
    difference_lines = []
    for item in comparison["differing_tasks"]:
        iou_text = ", ".join(f"{value:.6f}" for value in item["matched_ious"]) or "—"
        difference_lines.append(
            f"| `{Path(item['image']).name}` | "
            f"{', '.join(item['participant_1_classes']) or '—'} | "
            f"{', '.join(item['participant_2_classes']) or '—'} | {iou_text} |"
        )
    differences = (
        "| Изображение | Экспорт 1 | Экспорт 2 | IoU |\n"
        "| --- | --- | --- | ---: |\n"
        + "\n".join(difference_lines)
        if difference_lines
        else "Различий между итоговыми аннотациями не обнаружено."
    )
    errors = summary["errors"]
    result_text = (
        "Критических ошибок не обнаружено."
        if not errors
        else f"Обнаружено критических ошибок: {len(errors)}."
    )
    return f"""# Контроль качества ручной разметки HaGRID

Проверены два экспорта Label Studio для одинакового набора из 200 дополнительных изображений классов `like`, `dislike`, `palm` и `fist`. Для задач с повторными отправками анализатор выбирает последнюю активную непустую аннотацию.

| Показатель | Экспорт 1 | Экспорт 2 |
| --- | ---: | ---: |
| Задач | {first['tasks']} | {second['tasks']} |
| Уникальных изображений | {first['unique_images']} | {second['unique_images']} |
| Итоговых bbox | {first['selected_boxes']} | {second['selected_boxes']} |
| Задач без активной аннотации | {first['tasks_without_active_annotations']} | {second['tasks_without_active_annotations']} |
| Идентичных повторных отправок | {first['repeated_identical_submissions']} | {second['repeated_identical_submissions']} |

## Распределение классов

| Класс | Экспорт 1 | Экспорт 2 |
| --- | ---: | ---: |
{chr(10).join(rows)}

## Согласованность

| Метрика | Результат |
| --- | ---: |
| Совпадение классов | {comparison['single_label_class_matches']} из {comparison['single_label_tasks']} ({format_number(comparison['percentage_agreement'] * 100 if comparison['percentage_agreement'] is not None else None, 1)}%) |
| Cohen's Kappa | {format_number(comparison['cohens_kappa'])} |
| Точно совпавшие bbox | {comparison['exact_box_matches']} из {comparison['matched_boxes']} |
| Mean IoU | {format_number(comparison['iou_mean'], 6)} |
| Median IoU | {format_number(comparison['iou_median'], 6)} |
| Minimum IoU | {format_number(comparison['iou_min'], 6)} |

{differences}

Все выбранные bbox имеют допустимые классы и размеры, соответствуют изображениям и находятся в границах с учётом допуска на округление. {result_text}

## Ограничения интерпретации

Участники проверяли предварительную разметку, поэтому высокий IoU не равнозначен независимому построению bbox с нуля. Cohen's Kappa оценивает только совпадение классов, а не геометрию рамок. Технические поля экспортов не позволяют подтвердить личности участников или организационную независимость работы; это определяется протоколом проведения разметки.
"""


def main() -> int:
    args = parse_args()
    image_root = args.image_root.expanduser().resolve()
    image_size_cache: dict[str, tuple[int, int]] = {}
    first = parse_export(
        args.participant_1,
        image_root,
        args.expected_tasks,
        args.expected_per_class,
        args.bbox_tolerance_percent,
        image_size_cache,
    )
    second = parse_export(
        args.participant_2,
        image_root,
        args.expected_tasks,
        args.expected_per_class,
        args.bbox_tolerance_percent,
        image_size_cache,
    )
    comparison = compare_exports(first, second)
    errors = [f"participant_1: {item}" for item in first.errors] + [
        f"participant_2: {item}" for item in second.errors
    ]
    if comparison["only_participant_1"] or comparison["only_participant_2"]:
        errors.append("the two exports do not contain the same source_id set")
    warnings = [f"participant_1: {item}" for item in first.warnings] + [
        f"participant_2: {item}" for item in second.warnings
    ]
    warnings.append(
        "shared Label Studio IDs and timestamps do not prove annotator identity or independent work"
    )
    summary = {
        "status": "PASSED" if not errors else "FAILED",
        "protocol": {
            "expected_tasks": args.expected_tasks,
            "expected_per_class": args.expected_per_class,
            "allowed_labels": list(ALLOWED_LABELS),
            "bbox_tolerance_percent": args.bbox_tolerance_percent,
            "active_annotation_selection": "latest updated_at, then created_at, then JSON position",
            "iou_matching": "maximum total IoU within each class",
        },
        "exports": [first.summary, second.summary],
        "comparison": comparison,
        "errors": errors,
        "warnings": warnings,
    }

    output_json = args.output_json.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_report.write_text(render_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_json": display_path(output_json),
                "output_report": display_path(output_report),
                "tasks": [first.summary.get("tasks"), second.summary.get("tasks")],
                "class_matches": comparison["single_label_class_matches"],
                "matched_boxes": comparison["matched_boxes"],
                "differing_tasks": len(comparison["differing_tasks"]),
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
