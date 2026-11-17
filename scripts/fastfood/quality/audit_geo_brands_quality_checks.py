"""Проанализировать заполненный независимый контроль новой Geo-выборки.

До заполнения всех ручных полей скрипт сохраняет только статус ожидания и не
вычисляет качество слабых меток.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.fastfood.prepare_geo_brand_sample import DEFAULT_CONFIG, load_config
from scripts.fastfood.quality.prepare_geo_brands_quality_checks import (
    DEFAULT_ORIGINAL,
    DEFAULT_OUTPUT,
    DEFAULT_POOL,
    DEFAULT_SAMPLE,
    build_brand_names_review,
    dataframe_csv_bytes,
    select_brand_records,
    select_sentiment_reviews,
    write_bytes_if_changed,
)


ALLOWED_VALUES = {"0", "1", "ambiguous"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def read_manual_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def validate_manual_values(values: pd.Series, field: str) -> dict[str, int]:
    cleaned = values.fillna("").astype(str).str.strip()
    invalid = sorted(set(cleaned) - ALLOWED_VALUES - {""})
    if invalid:
        raise ValueError(f"Недопустимые значения {field}: {invalid}")
    return {
        "rows": len(cleaned),
        "filled": int(cleaned.ne("").sum()),
        "blank": int(cleaned.eq("").sum()),
        "ambiguous": int(cleaned.eq("ambiguous").sum()),
    }


def assert_template_integrity(
    annotation: pd.DataFrame,
    reference: pd.DataFrame,
    names_review: pd.DataFrame,
    records_review: pd.DataFrame,
    original: pd.DataFrame,
    pool: pd.DataFrame,
    sample: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    quality = config["quality_checks"]
    sentiment_quotas = {
        str(brand): int(count)
        for brand, count in quality["sentiment_train_quotas_per_label"].items()
    }
    brand_quotas = {
        str(brand): int(count) for brand, count in quality["brand_record_quotas"].items()
    }
    expected_annotation, expected_reference = select_sentiment_reviews(
        original, sentiment_quotas, int(quality["random_seed"]) + 50
    )
    expected_names = build_brand_names_review(pool)
    expected_records = select_brand_records(
        sample, brand_quotas, int(quality["random_seed"]) + 30
    )
    comparisons = (
        (
            annotation,
            expected_annotation,
            ["source_id", "text"],
            "weak_label_manual_check_50.csv",
        ),
        (
            reference,
            expected_reference,
            ["source_id", "brand", "rating", "weak_label", "split"],
            "weak_label_reference_50.csv",
        ),
        (
            names_review,
            expected_names,
            ["name_ru", "brand_assigned", "rubrics", "brand_match_rule", "review_count"],
            "brand_names_review.csv",
        ),
        (
            records_review,
            expected_records,
            ["source_id", "name_ru", "rubrics", "address", "brand_assigned"],
            "brand_records_review_30.csv",
        ),
    )
    for actual, expected, columns, filename in comparisons:
        actual_stable = actual[columns].fillna("").astype(str).reset_index(drop=True)
        expected_stable = expected[columns].fillna("").astype(str).reset_index(drop=True)
        if not actual_stable.equals(expected_stable):
            raise ValueError(f"Стабильные поля изменены в {filename}")


def analyze_sentiment(
    annotation: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    joined = annotation.merge(reference, on="source_id", validate="one_to_one")
    joined["manual_sentiment"] = joined["manual_sentiment"].str.strip()
    joined["comparison"] = "ambiguous"
    determined = joined["manual_sentiment"].isin(["0", "1"])
    joined.loc[determined, "comparison"] = "mismatch"
    manual_numeric = pd.to_numeric(joined.loc[determined, "manual_sentiment"])
    weak_numeric = pd.to_numeric(joined.loc[determined, "weak_label"])
    matches = manual_numeric.eq(weak_numeric)
    determined_index = joined.index[determined]
    joined.loc[determined_index[matches.to_numpy()], "comparison"] = "match"

    determined_frame = joined.loc[determined].copy()
    determined_frame["manual_numeric"] = determined_frame["manual_sentiment"].astype(int)
    determined_frame["weak_numeric"] = determined_frame["weak_label"].astype(int)
    matrix = pd.crosstab(
        determined_frame["manual_numeric"], determined_frame["weak_numeric"]
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    mismatch_frame = joined.loc[joined["comparison"].eq("mismatch")]
    summary = {
        "rows": len(joined),
        "determined": int(determined.sum()),
        "ambiguous": int((~determined).sum()),
        "matches": int(joined["comparison"].eq("match").sum()),
        "mismatches": int(joined["comparison"].eq("mismatch").sum()),
        "agreement_among_determined": (
            float(joined.loc[determined, "comparison"].eq("match").mean())
            if determined.any()
            else None
        ),
        "confusion_matrix_manual_rows_weak_columns": {
            str(row): {str(column): int(matrix.loc[row, column]) for column in (0, 1)}
            for row in (0, 1)
        },
        "mismatches_by_brand": {
            str(brand): int(count)
            for brand, count in mismatch_frame["brand"].value_counts().items()
        },
    }
    return joined, summary


def summarize_brand_checks(frame: pd.DataFrame, unit: str) -> dict[str, Any]:
    values = frame["manual_brand_check"].str.strip()
    return {
        "unit": unit,
        "rows": len(frame),
        "confirmed": int(values.eq("1").sum()),
        "rejected": int(values.eq("0").sum()),
        "ambiguous": int(values.eq("ambiguous").sum()),
        "issues_by_assigned_brand": {
            str(brand): {
                "rejected": int(part["manual_brand_check"].str.strip().eq("0").sum()),
                "ambiguous": int(
                    part["manual_brand_check"].str.strip().eq("ambiguous").sum()
                ),
            }
            for brand, part in frame.groupby("brand_assigned", sort=True)
        },
    }


def main() -> int:
    args = parse_args()
    root = args.quality_root.resolve()
    annotation = read_manual_csv(root / "weak_label_manual_check_50.csv")
    reference = read_manual_csv(root / "weak_label_reference_50.csv")
    names_review = read_manual_csv(root / "brand_names_review.csv")
    records_review = read_manual_csv(root / "brand_records_review_30.csv")
    original = pd.read_csv(args.original.resolve())
    pool = pd.read_parquet(args.pool.resolve())
    sample = pd.read_csv(args.sample.resolve())
    config = load_config(args.config.resolve())

    assert_template_integrity(
        annotation,
        reference,
        names_review,
        records_review,
        original,
        pool,
        sample,
        config,
    )
    progress = {
        "sentiment": validate_manual_values(
            annotation["manual_sentiment"], "manual_sentiment"
        ),
        "brand_names": validate_manual_values(
            names_review["manual_brand_check"], "brand_names.manual_brand_check"
        ),
        "brand_records": validate_manual_values(
            records_review["manual_brand_check"], "brand_records.manual_brand_check"
        ),
    }
    blanks = sum(section["blank"] for section in progress.values())
    if blanks:
        status = {
            "status": "NOT_COMPLETED",
            "message": "Ручная проверка ещё не выполнена полностью; метрики не рассчитаны.",
            "progress": progress,
        }
        write_bytes_if_changed(
            root / "audit_status.json",
            json.dumps(status, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 2

    comparison, sentiment_summary = analyze_sentiment(annotation, reference)
    report = {
        "status": "COMPLETED",
        "sentiment": sentiment_summary,
        "brand_names": summarize_brand_checks(names_review, "unique_name"),
        "brand_records": summarize_brand_checks(records_review, "record"),
    }
    write_bytes_if_changed(
        root / "sentiment_comparison_report.csv", dataframe_csv_bytes(comparison)
    )
    write_bytes_if_changed(
        root / "quality_audit_summary.json",
        json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    write_bytes_if_changed(
        root / "audit_status.json",
        json.dumps(
            {"status": "COMPLETED", "message": "Ручная проверка проанализирована."},
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
