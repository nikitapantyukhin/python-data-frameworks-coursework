"""Тесты подготовки ручного обогащения HaGRID."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.hagrid.manual_enrichment import (
    build_blank_task,
    hamming_distance,
    label_studio_config,
    load_config,
    load_selection_manifest,
)


class HagridManualEnrichmentTests(unittest.TestCase):
    def test_project_config_uses_current_source_and_mapping(self) -> None:
        config = load_config()
        self.assertEqual(
            config["source"]["dataset_url"],
            "https://huggingface.co/datasets/CNGR/CN_Gesture_Recognition",
        )
        self.assertEqual(config["source"]["license"], "MIT")
        self.assertEqual(
            config["class_mapping"],
            {
                "thumbs_up": "like",
                "thumbs_down": "dislike",
                "palm": "palm",
                "fist_up": "fist",
            },
        )

    def test_config_rejects_wrong_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "source": {"dataset_url": "https://example.invalid", "license": "MIT"},
                        "class_mapping": {
                            "thumbs_up": "like",
                            "thumbs_down": "dislike",
                            "palm": "palm",
                            "fist_up": "fist",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_config(path)

    def test_manifest_requires_reproducibility_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["task_id", "image_name"])
                writer.writeheader()
                writer.writerow({"task_id": 1, "image_name": "image_001.jpg"})
            with self.assertRaises(ValueError):
                load_selection_manifest(path)

    def test_hamming_distance_counts_different_bits(self) -> None:
        self.assertEqual(hamming_distance(0b1010, 0b0011), 2)

    def test_blank_task_contains_no_annotations_or_predictions(self) -> None:
        task = build_blank_task(1, "image_001.jpg", "hagrid_manual:001")
        self.assertEqual(task["data"]["image"], "/data/local-files/?d=images/image_001.jpg")
        self.assertNotIn("annotations", task)
        self.assertNotIn("predictions", task)
        json.dumps(task)

    def test_label_studio_config_has_all_four_classes(self) -> None:
        xml = label_studio_config()
        for label in ("like", "dislike", "palm", "fist"):
            self.assertIn(f'value="{label}"', xml)


if __name__ == "__main__":
    unittest.main()
