"""Проверить новую Geo-выборку, разбиения и режим Manual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.fastfood.fastfood_data import normalize_text, weak_label_from_rating
from scripts.fastfood.prepare_fastfood_brand_splits import frame_digest, prepare_manual
from scripts.fastfood.prepare_geo_brand_sample import (
    BRAND_MATCH_RULE,
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT as DEFAULT_SAMPLE_DIR,
    load_config,
    match_brand,
    sample_by_quotas,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = ROOT / "data" / "processed" / "fastfood_experiment"
DEFAULT_ANNOTATIONS = ROOT / ".local" / "fastfood_annotation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def assert_counts(actual: pd.Series, expected: dict[Any, int], context: str) -> None:
    normalized = {key: int(value) for key, value in actual.items()}
    if normalized != expected:
        raise ValueError(f"{context}: ожидалось {expected}, найдено {normalized}")


def validate_sample(
    pool: pd.DataFrame,
    sample: pd.DataFrame,
    config: dict[str, Any],
    manual_source: pd.DataFrame,
) -> dict[str, Any]:
    quotas = {
        str(brand): int(quota)
        for brand, quota in config["base_sample"]["sample_quotas_per_label"].items()
    }
    if len(sample) != 1000:
        raise ValueError(f"Основная выборка: ожидалось 1000 строк, найдено {len(sample)}")
    if pool["source_id"].duplicated().any() or pool["normalized_text"].duplicated().any():
        raise ValueError("Доступный пул содержит дубликаты")
    if pool["normalized_text"].eq("").any():
        raise ValueError("Доступный пул содержит пустой нормализованный текст")
    pool_labels = pool["rating"].map(weak_label_from_rating).astype(int)
    if not pool_labels.eq(pool["weak_label"]).all():
        raise ValueError("Слабые метки доступного пула не соответствуют рейтингам")
    assert_counts(sample["weak_label"].value_counts().sort_index(), {0: 500, 1: 500}, "Классы")
    for brand, quota in quotas.items():
        part = sample.loc[sample["brand"].eq(brand)]
        assert_counts(
            part["weak_label"].value_counts().sort_index(),
            {0: quota, 1: quota},
            f"Квота {brand}",
        )
    expected_labels = sample["rating"].map(weak_label_from_rating).astype(int)
    if not expected_labels.eq(sample["weak_label"]).all():
        raise ValueError("Слабые метки не соответствуют рейтингам")
    if sample["source_id"].duplicated().any():
        raise ValueError("Повторяющиеся source_id в основной выборке")
    if sample["normalized_text"].eq("").any() or sample["normalized_text"].duplicated().any():
        raise ValueError("Пустые или повторяющиеся нормализованные тексты")
    if not set(sample["source_id"]).issubset(set(pool["source_id"])):
        raise ValueError("Основная выборка содержит строки вне доступного пула")
    for row in sample[["name_ru", "rubrics", "brand", "brand_match_rule"]].itertuples(index=False):
        brand, rule = match_brand(row.name_ru, row.rubrics)
        if brand != row.brand or rule != row.brand_match_rule or rule != BRAND_MATCH_RULE:
            raise ValueError(f"Некорректное сопоставление бренда для {row.name_ru!r}")

    expected = sample_by_quotas(pool, quotas, int(config["random_seed"]))
    if expected["source_id"].tolist() != sample["source_id"].tolist():
        raise ValueError("Отбор не воспроизводится из пула с заданным seed")
    manual_normalized = set(manual_source["text"].map(normalize_text))
    overlap = set(sample["normalized_text"]) & manual_normalized
    if overlap:
        raise ValueError(f"Основная выборка пересекается с Otzyvru: {len(overlap)}")
    return {
        "rows": len(sample),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in sample["weak_label"].value_counts().sort_index().items()
        },
        "normalized_duplicates": int(sample["normalized_text"].duplicated().sum()),
        "manual_overlap": len(overlap),
        "deterministic_selection": True,
    }


def validate_splits(
    sample: pd.DataFrame,
    original: pd.DataFrame,
    combined: pd.DataFrame,
    config: dict[str, Any],
    annotation_root: Path,
) -> dict[str, Any]:
    if len(original) != 1000 or len(combined) != 1200:
        raise ValueError(f"Размеры Original/Manual: {len(original)}/{len(combined)}")
    assert_counts(
        original["split"].value_counts(),
        {"train": 700, "validation": 150, "test": 150},
        "Original split",
    )
    assert_counts(
        combined["split"].value_counts(),
        {"train": 900, "validation": 150, "test": 150},
        "Manual split",
    )
    for split, expected in {"train": 350, "validation": 75, "test": 75}.items():
        part = original.loc[original["split"].eq(split)]
        assert_counts(part["label"].value_counts().sort_index(), {0: expected, 1: expected}, f"{split} classes")

    split_quotas = config["split_quotas_per_brand_and_label"]
    for brand, quotas in split_quotas.items():
        for label in (0, 1):
            part = original.loc[
                original["brand"].eq(brand) & original["label"].eq(label)
            ]
            assert_counts(
                part["split"].value_counts(),
                {split: int(count) for split, count in quotas.items()},
                f"Split quota {brand}, label={label}",
            )
    if set(original["source_id"]) != set(sample["source_id"]):
        raise ValueError("Original не совпадает с основной выборкой по source_id")
    if original["source_id"].duplicated().any() or original["normalized_text"].duplicated().any():
        raise ValueError("В Original найдены дубликаты")
    split_sets = {
        split: set(part["source_id"])
        for split, part in original.groupby("split", sort=False)
    }
    norm_sets = {
        split: set(part["normalized_text"])
        for split, part in original.groupby("split", sort=False)
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_sets[left] & split_sets[right] or norm_sets[left] & norm_sets[right]:
            raise ValueError(f"Пересечение между {left} и {right}")

    geo_combined = combined.loc[combined["source_type"].eq("geo_reviews")].copy()
    manual_combined = combined.loc[combined["source_type"].eq("otzyvru")].copy()
    if len(geo_combined) != 1000 or len(manual_combined) != 200:
        raise ValueError("Неверное число строк источников в Manual")
    if not manual_combined["split"].eq("train").all():
        raise ValueError("Otzyvru найден вне train")
    if set(geo_combined["source_id"]) != set(original["source_id"]):
        raise ValueError("Geo-часть Manual отличается от Original")

    for split in ("validation", "test"):
        left = original.loc[original["split"].eq(split)]
        right = combined.loc[combined["split"].eq(split)]
        if frame_digest(left) != frame_digest(right):
            raise ValueError(f"{split} различается между Original и Manual")

    expected_manual = prepare_manual(annotation_root)
    expected_manual = expected_manual.sort_values("source_id").reset_index(drop=True)
    actual_manual = manual_combined.sort_values("source_id").reset_index(drop=True)
    for column in ("source_id", "text", "label", "brand", "normalized_text", "split"):
        if expected_manual[column].tolist() != actual_manual[column].tolist():
            raise ValueError(f"Ручные отзывы изменены в поле {column}")
    assert_counts(
        manual_combined["brand"].value_counts(),
        {"KFC": 70, "Burger King": 70, "McDonald's": 60},
        "Бренды Otzyvru",
    )
    assert_counts(manual_combined["label"].value_counts().sort_index(), {0: 100, 1: 100}, "Метки Otzyvru")
    if set(original["normalized_text"]) & set(manual_combined["normalized_text"]):
        raise ValueError("Нормализованное пересечение Geo/Otzyvru")
    return {
        "original_rows": len(original),
        "manual_rows": len(combined),
        "original_split_counts": {
            str(split): int(count) for split, count in original["split"].value_counts().items()
        },
        "manual_split_counts": {
            str(split): int(count) for split, count in combined["split"].value_counts().items()
        },
        "manual_rows_preserved": len(manual_combined),
        "validation_identical": True,
        "test_identical": True,
        "cross_source_overlap": 0,
    }


def main() -> int:
    args = parse_args()
    sample_dir = args.sample_dir.resolve()
    experiment_dir = args.experiment_dir.resolve()
    config = load_config(args.config.resolve())
    selection_metadata = json.loads(
        (sample_dir / "selection_summary.json").read_text(encoding="utf-8")
    )
    split_metadata = json.loads(
        (experiment_dir / "split_summary.json").read_text(encoding="utf-8")
    )
    pool = pd.read_parquet(sample_dir / "available_pool.parquet")
    sample = pd.read_csv(sample_dir / "sample_1000.csv")
    original = pd.read_csv(experiment_dir / "original.csv")
    combined = pd.read_csv(experiment_dir / "manual.csv")
    manual_source = pd.read_csv(args.annotation_root.resolve() / "04_final_dataset.csv")
    sample_report = validate_sample(pool, sample, config, manual_source)
    split_report = validate_splits(
        sample, original, combined, config, args.annotation_root.resolve()
    )
    if selection_metadata.get("eligible_rows") != len(pool):
        raise ValueError("selection_summary.json не соответствует доступному пулу")
    if selection_metadata.get("sample_rows") != len(sample):
        raise ValueError("selection_summary.json не соответствует основной выборке")
    if split_metadata.get("original_rows") != len(original):
        raise ValueError("split_summary.json не соответствует Original")
    if split_metadata.get("manual_rows") != len(combined):
        raise ValueError("split_summary.json не соответствует Manual")
    report = {
        "status": "PASSED",
        "sample": sample_report,
        "splits": split_report,
        "metadata_files_consistent": True,
    }
    output = experiment_dir / "validation_summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
