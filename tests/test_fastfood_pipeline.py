"""Базовые тесты общих правил первого датасета."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.fastfood.fastfood_data import normalize_text, rubric_set, weak_label_from_rating
from scripts.fastfood.quality.audit_fastfood_annotations import cohens_kappa


class FastfoodPipelineTests(unittest.TestCase):
    def test_normalization_collapses_case_punctuation_and_spaces(self) -> None:
        self.assertEqual(normalize_text("  Очень,  ВКУСНО!!! "), "очень вкусно")

    def test_rubric_parsing_uses_exact_components(self) -> None:
        self.assertEqual(
            rubric_set("Кафе; Быстрое питание;Ресторан"),
            {"Кафе", "Быстрое питание", "Ресторан"},
        )

    def test_rating_to_weak_label_policy(self) -> None:
        self.assertEqual(
            [
                weak_label_from_rating(value)
                for value in (0, 1, 2, 3, 4, 5, None, pd.NA)
            ],
            [-1, 0, 0, -1, 1, 1, -1, -1],
        )

    def test_kappa_uses_independent_labels(self) -> None:
        left = pd.Series([0, 0, 1, 1])
        right = pd.Series([0, 1, 1, 1])
        self.assertAlmostEqual(cohens_kappa(left, right), 0.5)


if __name__ == "__main__":
    unittest.main()
