"""Синтетические тесты четырёхбрендового конвейера Geo Reviews."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.fastfood.prepare_fastfood_brand_splits import assign_brand_splits
from scripts.fastfood.prepare_geo_brand_sample import (
    BRAND_MATCH_RULE,
    build_eligible_pool,
    match_brand,
    sample_by_quotas,
)


class GeoBrandPipelineTests(unittest.TestCase):
    def test_brand_whitelist_requires_exact_rubric(self) -> None:
        self.assertEqual(
            match_brand("KFC Авто", "Быстрое питание;Кафе"),
            ("KFC", BRAND_MATCH_RULE),
        )
        self.assertEqual(
            match_brand("ROSTIC'S АВТО", "Быстрое питание"),
            ("Rostic's", BRAND_MATCH_RULE),
        )
        self.assertEqual(match_brand("БК", "Быстрое питание"), (None, None))
        self.assertEqual(match_brand("KFC", "Товары для здоровья"), (None, None))
        self.assertEqual(match_brand("McDonald's", "Быстрое питание"), (None, None))

    def test_pool_removes_label_conflicts_cross_brand_and_duplicates(self) -> None:
        frame = pd.DataFrame(
            {
                "address": [f"a{i}" for i in range(9)],
                "name_ru": [
                    "KFC",
                    "KFC Авто",
                    "KFC",
                    "Бургер Кинг",
                    "KFC",
                    "Бургер Кинг",
                    "Rostic's",
                    "Вкусно — и точка",
                    "KFC",
                ],
                "rubrics": ["Быстрое питание"] * 8 + ["Кафе"],
                "rating": [1, 1, 1, 5, 1, 1, 1, 3, 1],
                "text": [
                    "unique kfc",
                    "same duplicate",
                    "same duplicate!",
                    "label conflict",
                    "label conflict",
                    "cross brand text",
                    "cross brand text",
                    "excluded rating",
                    "wrong rubric",
                ],
            }
        )
        pool, summary = build_eligible_pool(frame)
        self.assertEqual(pool["text"].tolist(), ["unique kfc", "same duplicate"])
        self.assertEqual(summary["normalized_duplicate_extra"], 1)
        self.assertEqual(summary["label_conflict_groups"], 1)
        self.assertEqual(summary["cross_brand_duplicate_groups"], 2)

    def test_quota_sample_is_exact_and_deterministic(self) -> None:
        rows = []
        for brand in ("KFC", "Burger King"):
            for label in (0, 1):
                for index in range(8):
                    rows.append(
                        {
                            "source_id": f"{brand}:{label}:{index}",
                            "brand": brand,
                            "weak_label": label,
                        }
                    )
        pool = pd.DataFrame(rows)
        quotas = {"KFC": 3, "Burger King": 2}
        first = sample_by_quotas(pool, quotas, 42)
        second = sample_by_quotas(pool, quotas, 42)
        self.assertEqual(first["source_id"].tolist(), second["source_id"].tolist())
        self.assertEqual(
            pd.crosstab(first["brand"], first["weak_label"]).to_dict(),
            {0: {"Burger King": 2, "KFC": 3}, 1: {"Burger King": 2, "KFC": 3}},
        )

    def test_brand_split_quotas_are_exact(self) -> None:
        rows = []
        for label in (0, 1):
            for index in range(6):
                rows.append(
                    {
                        "source_id": f"KFC:{label}:{index}",
                        "brand": "KFC",
                        "weak_label": label,
                    }
                )
        result = assign_brand_splits(
            pd.DataFrame(rows),
            {"KFC": {"train": 4, "validation": 1, "test": 1}},
            42,
        )
        for label in (0, 1):
            counts = result.loc[result["weak_label"].eq(label), "split"].value_counts()
            self.assertEqual(counts.to_dict(), {"train": 4, "test": 1, "validation": 1})


if __name__ == "__main__":
    unittest.main()
