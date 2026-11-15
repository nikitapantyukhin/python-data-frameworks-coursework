"""Небольшие синтетические проверки подготовки первого датасета."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

from scripts.fastfood.fastfood_data import (
    assign_splits,
    balanced_sample,
    csv_export_frame,
    ensure_disjoint,
    normalize_text,
    rubric_set,
    serialize_rubric_set,
    stratified_sample,
    weak_label_from_rating,
)
from scripts.fastfood.quality.audit_fastfood_annotations import cohens_kappa


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
            "from scripts.fastfood.fastfood_data import serialize_rubric_set; "
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

    def test_rating_to_weak_label_policy(self) -> None:
        self.assertEqual(
            [
                weak_label_from_rating(value)
                for value in (0, 1, 2, 3, 4, 5, None, pd.NA)
            ],
            [-1, 0, 0, -1, 1, 1, -1, -1],
        )

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
