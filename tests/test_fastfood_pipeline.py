"""Небольшие синтетические проверки подготовки первого датасета."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.fastfood.fastfood_data import ensure_disjoint, normalize_text
from scripts.fastfood.prepare_fastfood_splits import assign_splits
from scripts.fastfood.prepare_geo_fastfood import (
    balanced_sample,
    csv_export_frame,
    rubric_set,
    serialize_rubric_set,
)
from scripts.fastfood.quality.audit_fastfood_annotations import cohens_kappa
from scripts.fastfood.quality.prepare_geo_quality_checks import stratified_sample


ROOT = Path(__file__).resolve().parents[1]


class FastfoodPipelineTests(unittest.TestCase):
    def test_normalization_collapses_case_punctuation_and_spaces(self) -> None:
        self.assertEqual(normalize_text("  Очень,  ВКУСНО!!! "), "очень вкусно")

    def test_rubric_parsing_uses_exact_components(self) -> None:
        self.assertEqual(
            rubric_set("Кафе; Быстрое питание;Ресторан"),
            {"Кафе", "Быстрое питание", "Ресторан"},
        )

    def test_rubric_serialization_is_sorted_without_mutating_source(self) -> None:
        rubrics = {"Ресторан", "Быстрое питание", "Кафе"}
        frame = pd.DataFrame({"rubric_set": [rubrics], "value": [1]})
        exported = csv_export_frame(frame)
        self.assertEqual(
            exported.loc[0, "rubric_set"],
            "Быстрое питание;Кафе;Ресторан",
        )
        self.assertIsInstance(frame.loc[0, "rubric_set"], set)

    def test_rubric_serialization_ignores_python_hash_seed(self) -> None:
        code = (
            "from scripts.fastfood.prepare_geo_fastfood import serialize_rubric_set; "
            "print(serialize_rubric_set({'Ресторан', 'Быстрое питание', 'Кафе'}))"
        )
        outputs = []
        for hash_seed in ("1", "2"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            environment["PYTHONPATH"] = str(ROOT)
            outputs.append(
                subprocess.check_output(
                    [sys.executable, "-c", code],
                    env=environment,
                    text=True,
                )
            )
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(
            outputs[0].strip(),
            serialize_rubric_set(rubric_set("Кафе;Ресторан;Быстрое питание")),
        )

    def test_eda_summary_output_can_be_redirected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            geo_sample = temporary / "geo.csv"
            summary_output = temporary / "nested" / "summary.json"
            pd.DataFrame(
                {
                    "weak_label": [0, 1],
                    "rating": [1, 5],
                    "text": [
                        "synthetic negative fixture",
                        "synthetic positive fixture",
                    ],
                }
            ).to_csv(geo_sample, index=False)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.fastfood.eda_fastfood",
                    "--geo-sample",
                    str(geo_sample),
                    "--annotation-root",
                    str(temporary / "missing_annotations"),
                    "--figures-dir",
                    str(temporary / "figures"),
                    "--summary-output",
                    str(summary_output),
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            self.assertTrue(summary_output.is_file())
            summary = json.loads(summary_output.read_text(encoding="utf-8"))
            self.assertEqual(summary["geo_rows"], 2)
            self.assertEqual(summary["manual_status"], "not_available")

    def test_balanced_sample_is_reproducible(self) -> None:
        frame = pd.DataFrame(
            {
                "source_id": [f"id:{i}" for i in range(20)],
                "weak_label": [0] * 10 + [1] * 10,
            }
        )
        first = balanced_sample(frame, 10, 42)
        second = balanced_sample(frame, 10, 42)
        self.assertEqual(first["source_id"].tolist(), second["source_id"].tolist())
        self.assertEqual(first["weak_label"].value_counts().to_dict(), {0: 5, 1: 5})

    def test_kappa_uses_independent_labels(self) -> None:
        left = pd.Series([0, 0, 1, 1])
        right = pd.Series([0, 1, 1, 1])
        self.assertAlmostEqual(cohens_kappa(left, right), 0.5)

    def test_group_split_has_no_normalized_leakage(self) -> None:
        frame = pd.DataFrame(
            {
                "label": [0] * 10 + [1] * 10,
                "group_id": [f"g{i}" for i in range(20)],
                "text": [f"synthetic fixture {i}" for i in range(20)],
            }
        )
        split = assign_splits(frame, 42, 0.2, 0.2)
        ensure_disjoint(
            (name, part["text"]) for name, part in split.groupby("split", sort=True)
        )
        self.assertEqual(split["split"].value_counts().to_dict(), {"train": 12, "test": 4, "validation": 4})

    def test_leakage_guard_rejects_cross_split_duplicate(self) -> None:
        with self.assertRaises(ValueError):
            ensure_disjoint(
                [("train", ["Один текст"]), ("test", ["один, текст!"])]
            )

    def test_quality_sample_respects_split_and_class_quotas(self) -> None:
        frame = pd.DataFrame(
            {
                "source_id": [f"id:{i}" for i in range(24)],
                "split": ["train"] * 8 + ["validation"] * 8 + ["test"] * 8,
                "label": ([0] * 4 + [1] * 4) * 3,
            }
        )
        quotas = {
            "train": {0: 2, 1: 2},
            "validation": {0: 1, 1: 1},
            "test": {0: 1, 1: 1},
        }
        first = stratified_sample(frame, quotas, 42)
        second = stratified_sample(frame, quotas, 42)
        self.assertEqual(first["source_id"].tolist(), second["source_id"].tolist())
        self.assertEqual(
            pd.crosstab(first["split"], first["label"]).to_dict(),
            {0: {"test": 1, "train": 2, "validation": 1},
             1: {"test": 1, "train": 2, "validation": 1}},
        )


if __name__ == "__main__":
    unittest.main()
