"""Подготовить общие original/manual разбиения без обучения моделей.

Синтетические режимы намеренно не создаются, пока отсутствуют реальные
синтетические данные. Дополнительные ручные отзывы добавляются только в train.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.fastfood.fastfood_data import ensure_disjoint, normalize_text, stable_hash
from scripts.fastfood.quality.audit_fastfood_annotations import canonical, locate_bundle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINAL = ROOT / "data" / "processed" / "geo_fastfood" / "sample_1000.csv"
DEFAULT_ANNOTATIONS = ROOT / ".local" / "fastfood_annotation"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "fastfood_experiment"
SEED = 20261118


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    return parser.parse_args()


def assign_splits(
    frame: pd.DataFrame, seed: int, validation_fraction: float, test_fraction: float
) -> pd.DataFrame:
    result = frame.copy()
    result["split"] = "train"
    for label, subset in result.groupby("label", sort=True):
        ordered = subset.assign(
            _key=subset["group_id"].map(lambda value: stable_hash(value, seed))
        ).sort_values("_key")
        count = len(ordered)
        test_count = round(count * test_fraction)
        validation_count = round(count * validation_fraction)
        result.loc[ordered.index[:test_count], "split"] = "test"
        result.loc[
            ordered.index[test_count : test_count + validation_count], "split"
        ] = "validation"
    return result


def main() -> int:
    args = parse_args()
    if args.validation_fraction + args.test_fraction >= 1:
        raise ValueError("Сумма долей validation и test должна быть меньше 1")
    original = pd.read_csv(args.original)
    required = {"source_id", "text", "weak_label"}
    missing = required - set(original.columns)
    if missing:
        raise ValueError(f"В original отсутствуют столбцы: {sorted(missing)}")
    original = original.rename(columns={"weak_label": "label"}).copy()
    original["source_type"] = "geo_reviews"
    original["label_type"] = "weak_rating"
    original["normalized_text"] = original["text"].map(normalize_text)
    original["group_id"] = original["normalized_text"].map(
        lambda value: stable_hash(value, 0)
    )
    if original["normalized_text"].duplicated().any():
        raise ValueError("Original содержит нормализованные дубликаты")
    original = assign_splits(
        original,
        args.seed,
        args.validation_fraction,
        args.test_fraction,
    )
    ensure_disjoint(
        (name, part["text"])
        for name, part in original.groupby("split", sort=True)
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original.to_csv(output / "original.csv", index=False)
    summary: dict = {
        "seed": args.seed,
        "normalization": "NFKC + casefold + punctuation/whitespace collapse",
        "original_rows": len(original),
        "original_splits": {
            str(key): int(value) for key, value in original["split"].value_counts().items()
        },
        "manual_status": "not_available",
        "synthetic_status": "not_created",
        "combined_status": "not_created",
    }

    try:
        bundle = locate_bundle(args.annotation_root)
    except FileNotFoundError:
        bundle = None
    if bundle is not None:
        manual = canonical(pd.read_csv(bundle / "04_final_dataset.csv"))
        manual = manual.rename(columns={"id": "manual_id"})
        manual["source_id"] = "otzyvru:" + manual["manual_id"].astype(str)
        manual["source_type"] = "otzyvru"
        manual["label_type"] = "independent_manual_consensus"
        manual["normalized_text"] = manual["text"].map(normalize_text)
        manual["group_id"] = manual["normalized_text"].map(
            lambda value: stable_hash(value, 0)
        )
        manual["split"] = "train"
        overlap = set(original["normalized_text"]) & set(manual["normalized_text"])
        if overlap:
            raise ValueError(
                f"Межисточниковая утечка: {len(overlap)} нормализованных пересечений"
            )
        if manual["normalized_text"].duplicated().any():
            raise ValueError("Manual содержит нормализованные дубликаты")
        combined_manual = pd.concat([original, manual], ignore_index=True, sort=False)
        ensure_disjoint(
            (name, part["text"])
            for name, part in combined_manual.groupby("split", sort=True)
        )
        combined_manual.to_csv(output / "manual.csv", index=False)
        summary["manual_status"] = "prepared_train_only"
        summary["manual_rows_added"] = len(manual)

    (output / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
