"""Тесты независимого контроля новой четырёхбрендовой выборки."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.fastfood.quality.audit_geo_brands_quality_checks import (
    analyze_sentiment,
    validate_manual_values,
)
from scripts.fastfood.quality.prepare_geo_brands_quality_checks import (
    assert_no_manual_answers,
    dataframe_csv_bytes,
    find_fuzzy_overlaps,
    safe_write_template,
    select_sentiment_reviews,
    write_bytes_if_changed,
)


class GeoBrandsQualityChecksTests(unittest.TestCase):
    def synthetic_original(self) -> pd.DataFrame:
        rows = []
        for brand in ("KFC", "Burger King"):
            for label in (0, 1):
                for index in range(6):
                    rows.append(
                        {
                            "source_id": f"{brand}:{label}:{index}",
                            "text": f"review {brand} {label} {index}",
                            "brand": brand,
                            "rating": 1 if label == 0 else 5,
                            "weak_label": label,
                            "split": "validation" if index == 5 else "train",
                        }
                    )
        return pd.DataFrame(rows)

    def test_sentiment_selection_is_blind_train_only_and_reproducible(self) -> None:
        original = self.synthetic_original()
        quotas = {"KFC": 2, "Burger King": 1}
        annotation, reference = select_sentiment_reviews(original, quotas, 42)
        second_annotation, second_reference = select_sentiment_reviews(original, quotas, 42)
        self.assertEqual(len(annotation), 6)
        self.assertEqual(
            list(annotation.columns),
            ["source_id", "text", "manual_sentiment", "manual_notes"],
        )
        self.assertTrue(annotation["manual_sentiment"].eq("").all())
        self.assertTrue(annotation["manual_notes"].eq("").all())
        self.assertEqual(set(annotation["source_id"]), set(reference["source_id"]))
        self.assertTrue(reference["split"].eq("train").all())
        self.assertEqual(annotation["source_id"].tolist(), second_annotation["source_id"].tolist())
        self.assertEqual(reference["source_id"].tolist(), second_reference["source_id"].tolist())
        counts = reference.groupby(["brand", "weak_label"]).size().to_dict()
        self.assertEqual(
            counts,
            {("Burger King", 0): 1, ("Burger King", 1): 1, ("KFC", 0): 2, ("KFC", 1): 2},
        )

    def test_manual_values_and_ambiguous_analysis(self) -> None:
        self.assertEqual(
            validate_manual_values(pd.Series(["0", "1", "ambiguous", ""]), "field"),
            {"rows": 4, "filled": 3, "blank": 1, "ambiguous": 1},
        )
        with self.assertRaises(ValueError):
            validate_manual_values(pd.Series(["positive"]), "field")
        annotation = pd.DataFrame(
            {
                "source_id": ["a", "b", "c"],
                "text": ["a", "b", "c"],
                "manual_sentiment": ["0", "1", "ambiguous"],
                "manual_notes": ["", "", ""],
            }
        )
        reference = pd.DataFrame(
            {
                "source_id": ["a", "b", "c"],
                "brand": ["KFC", "KFC", "Burger King"],
                "rating": [1, 1, 5],
                "weak_label": [0, 0, 1],
                "split": ["train"] * 3,
            }
        )
        comparison, summary = analyze_sentiment(annotation, reference)
        self.assertEqual(comparison["comparison"].tolist(), ["match", "mismatch", "ambiguous"])
        self.assertEqual(summary["matches"], 1)
        self.assertEqual(summary["mismatches"], 1)
        self.assertEqual(summary["ambiguous"], 1)
        self.assertEqual(summary["agreement_among_determined"], 0.5)

    def test_existing_manual_answers_are_not_overwritten(self) -> None:
        template = pd.DataFrame(
            {
                "source_id": ["a"],
                "text": ["text"],
                "manual_sentiment": [""],
                "manual_notes": [""],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "weak_label_manual_check_50.csv"
            self.assertEqual(
                safe_write_template(path, template, ("manual_sentiment", "manual_notes")),
                "created",
            )
            filled = template.copy()
            filled.loc[0, "manual_sentiment"] = "1"
            path.write_bytes(dataframe_csv_bytes(filled))
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                assert_no_manual_answers(root)
            self.assertEqual(path.read_bytes(), before)

    def test_fuzzy_duplicates_are_detected_and_categorized(self) -> None:
        records = pd.DataFrame(
            {
                "source_id": ["geo:a", "geo:b", "otzyvru:1", "otzyvru:2"],
                "text": [
                    "Очень вкусная еда и быстрое вежливое обслуживание сегодня",
                    "Очень вкусная еда и быстрое вежливое обслуживание вчера",
                    "Прекрасный ресторан и очень внимательные сотрудники",
                    "Прекрасный ресторан и очень внимательные сотрудники",
                ],
                "split": ["train", "validation", "train", "train"],
                "source": ["geo_reviews", "geo_reviews", "otzyvru", "otzyvru"],
            }
        )
        records["normalized_text"] = records["text"].str.casefold()
        records["text_length"] = records["text"].str.len()
        report, summary = find_fuzzy_overlaps(records, threshold=0.88)
        self.assertEqual(len(report), 2)
        self.assertEqual(summary["counts_by_category"]["train_vs_validation"], 1)
        self.assertEqual(summary["counts_by_category"]["within_otzyvru"], 1)

    def test_report_writer_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            self.assertEqual(write_bytes_if_changed(path, b"{}"), "written")
            self.assertEqual(write_bytes_if_changed(path, b"{}"), "unchanged")
            self.assertEqual(path.read_bytes(), b"{}")


if __name__ == "__main__":
    unittest.main()
