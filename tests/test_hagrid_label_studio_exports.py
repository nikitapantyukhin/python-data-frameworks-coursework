"""Тесты анализа двух экспортов Label Studio для ручного обогащения HaGRID."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.hagrid.analyze_label_studio_exports import (
    Box,
    bbox_iou,
    best_box_matching,
    cohens_kappa,
    compare_exports,
    parse_export,
)


def result(x: float = 10.0, label: str = "like") -> dict[str, object]:
    return {
        "original_width": 100,
        "original_height": 80,
        "image_rotation": 0,
        "value": {
            "x": x,
            "y": 20.0,
            "width": 30.0,
            "height": 40.0,
            "rotation": 0,
            "rectanglelabels": [label],
        },
        "id": "source_box_1",
        "from_name": "gesture",
        "to_name": "image",
        "type": "rectanglelabels",
        "origin": "prediction",
    }


def annotation(identifier: int, value: dict[str, object], updated_at: str) -> dict[str, object]:
    return {
        "id": identifier,
        "completed_by": 1,
        "created_at": updated_at,
        "updated_at": updated_at,
        "was_cancelled": False,
        "result": [value],
        "prediction": {"result": [result()]},
    }


def task(values: list[dict[str, object]]) -> dict[str, object]:
    return {
        "id": 1,
        "project": 5,
        "data": {
            "source_id": "hagrid_manual:001",
            "image": "/data/local-files/?d=images/image_001.jpg",
        },
        "annotations": values,
    }


class HagridLabelStudioExportTests(unittest.TestCase):
    def test_parse_export_collapses_identical_resubmissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            image_root.mkdir()
            Image.new("RGB", (100, 80)).save(image_root / "image_001.jpg")
            first = annotation(1, result(), "2026-11-16T10:00:00Z")
            second = annotation(2, result(), "2026-11-16T10:01:00Z")
            export_path = root / "export.json"
            export_path.write_text(json.dumps([task([first, second])]), encoding="utf-8")

            parsed = parse_export(export_path, image_root, 1, 1, 1e-4, {})

        self.assertEqual(parsed.errors, [])
        self.assertEqual(parsed.summary["active_annotations"], 2)
        self.assertEqual(parsed.summary["selected_annotations"], 1)
        self.assertEqual(parsed.summary["tasks_with_multiple_active_annotations"], 1)
        self.assertEqual(parsed.summary["repeated_identical_submissions"], 1)
        self.assertEqual(parsed.summary["repeated_divergent_variants"], 0)
        self.assertEqual(parsed.tasks["hagrid_manual:001"].annotation_id, 2)

    def test_compare_exports_reports_geometry_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            image_root.mkdir()
            Image.new("RGB", (100, 80)).save(image_root / "image_001.jpg")
            paths = []
            for index, x in enumerate((10.0, 15.0), 1):
                path = root / f"export_{index}.json"
                path.write_text(
                    json.dumps(
                        [
                            task(
                                [
                                    annotation(
                                        index,
                                        result(x=x),
                                        f"2026-11-16T10:0{index}:00Z",
                                    )
                                ]
                            )
                        ]
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            cache: dict[str, tuple[int, int]] = {}
            left = parse_export(paths[0], image_root, 1, 1, 1e-4, cache)
            right = parse_export(paths[1], image_root, 1, 1, 1e-4, cache)
            comparison = compare_exports(left, right)

        self.assertEqual(comparison["single_label_class_matches"], 1)
        self.assertEqual(comparison["cohens_kappa"], 1.0)
        self.assertEqual(comparison["exact_box_matches"], 0)
        self.assertEqual(len(comparison["differing_tasks"]), 1)
        self.assertAlmostEqual(comparison["iou_min"], 5 / 7)

    def test_bbox_matching_and_kappa(self) -> None:
        left = [Box("palm", 0, 0, 20, 20, 100, 100, 0, 0)]
        right = [Box("palm", 0, 0, 20, 20, 100, 100, 0, 0)]
        pairs, missing_left, missing_right = best_box_matching(left, right)
        self.assertEqual(pairs, [(0, 0, 1.0)])
        self.assertEqual((missing_left, missing_right), (0, 0))
        self.assertEqual(bbox_iou(left[0], right[0]), 1.0)
        self.assertAlmostEqual(
            cohens_kappa(["like", "like", "fist", "fist"], ["like", "fist", "fist", "fist"]),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
