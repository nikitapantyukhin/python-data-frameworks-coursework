"""Создать Original/Manual разбиения для четырёхбрендовой Geo-выборки."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.fastfood.fastfood_data import normalize_text, stable_hash
from scripts.fastfood.prepare_geo_brand_sample import DEFAULT_CONFIG, load_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE = ROOT / "data" / "processed" / "geo_fastfood_brands" / "sample_1000.csv"
DEFAULT_ANNOTATIONS = ROOT / ".local" / "fastfood_annotation"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "fastfood_experiment"
SPLIT_ORDER = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def assign_brand_splits(
    sample: pd.DataFrame,
    quotas: dict[str, dict[str, int]],
    seed: int,
) -> pd.DataFrame:
    """Назначить точные split-квоты отдельно для каждого бренда и класса."""

    result = sample.copy()
    result["split"] = pd.NA
    for brand, split_quotas in quotas.items():
        expected = sum(int(split_quotas[name]) for name in SPLIT_ORDER)
        for label in (0, 1):
            subset = result.loc[
                result["brand"].eq(brand) & result["weak_label"].eq(label)
            ].copy()
            if len(subset) != expected:
                raise ValueError(
                    f"Для {brand}, label={label} найдено {len(subset)}, "
                    f"для разбиения требуется {expected}"
                )
            subset["_split_key"] = subset["source_id"].map(
                lambda value: stable_hash(f"split\0{brand}\0{label}\0{value}", seed)
            )
            ordered = subset.sort_values("_split_key")
            start = 0
            for split in SPLIT_ORDER:
                stop = start + int(split_quotas[split])
                result.loc[ordered.index[start:stop], "split"] = split
                start = stop
    if result["split"].isna().any():
        missing = result.loc[result["split"].isna(), "source_id"].tolist()
        raise ValueError(f"Не назначен split для строк: {missing[:10]}")
    return result


def manual_brand(value: object) -> str:
    mapping = {
        "kfc": "KFC",
        "burger king": "Burger King",
        "mcdonald s": "McDonald's",
        "mcdonalds": "McDonald's",
    }
    key = normalize_text(value)
    if key not in mapping:
        raise ValueError(f"Неизвестный бренд Otzyvru: {value!r}")
    return mapping[key]


def prepare_manual(annotation_root: Path) -> pd.DataFrame:
    """Прочитать согласованные метки и provenance без изменения источников."""

    final = pd.read_csv(annotation_root / "04_final_dataset.csv")
    provenance = pd.read_csv(annotation_root / "05_provenance.csv")
    required_final = {"id", "text", "label"}
    required_provenance = {
        "id",
        "restaurant",
        "source_site",
        "source_page_url",
        "source_row",
        "source_rating",
        "collection_date",
    }
    if missing := required_final - set(final.columns):
        raise ValueError(f"В 04_final_dataset.csv отсутствуют поля: {sorted(missing)}")
    if missing := required_provenance - set(provenance.columns):
        raise ValueError(f"В 05_provenance.csv отсутствуют поля: {sorted(missing)}")
    merged = final.merge(provenance, on="id", validate="one_to_one")
    if len(merged) != 200 or merged["id"].nunique() != 200:
        raise ValueError("Ожидались 200 уникальных ручных отзывов")
    if not set(merged["label"]).issubset({0, 1}):
        raise ValueError("Ручные метки должны быть равны 0 или 1")

    result = pd.DataFrame(
        {
            "source_id": "otzyvru:" + merged["id"].astype(str),
            "address": pd.NA,
            "name_ru": merged["restaurant"],
            "rubrics": pd.NA,
            "rating": merged["source_rating"],
            "text": merged["text"],
            "brand": merged["restaurant"].map(manual_brand),
            "brand_match_rule": "otzyvru_provenance.restaurant",
            "weak_label": pd.NA,
            "label": merged["label"].astype("int8"),
            "label_type": "independent_manual_consensus",
            "source_type": "otzyvru",
            "split": "train",
            "manual_id": merged["id"].astype(str),
            "source_site": merged["source_site"],
            "source_page_url": merged["source_page_url"],
            "source_row": merged["source_row"],
            "collection_date": merged["collection_date"],
        }
    )
    result["normalized_name"] = result["name_ru"].map(normalize_text)
    result["normalized_text"] = result["text"].map(normalize_text)
    result["char_len"] = result["text"].str.len()
    result["word_len"] = result["text"].str.count(r"\S+")
    result["group_id"] = result["normalized_text"].map(
        lambda value: stable_hash(value, 0)
    )
    if result["normalized_text"].eq("").any():
        raise ValueError("В Otzyvru найден пустой нормализованный текст")
    if result["normalized_text"].duplicated().any():
        raise ValueError("В Otzyvru найдены нормализованные дубликаты")
    return result


def frame_digest(frame: pd.DataFrame) -> str:
    """Получить стабильный digest ключевых значений набора."""

    columns = ["source_id", "text", "label", "brand", "normalized_text", "split"]
    ordered = frame[columns].sort_values("source_id")
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def nested_counts(frame: pd.DataFrame, value: str) -> dict[str, Any]:
    return {
        str(brand): {
            str(int(label)): {
                str(split): int(count)
                for split, count in label_part[value].value_counts().sort_index().items()
            }
            for label, label_part in brand_part.groupby("label", sort=True)
        }
        for brand, brand_part in frame.groupby("brand", sort=False)
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    seed = int(config["random_seed"])
    quotas = {
        str(brand): {str(split): int(count) for split, count in values.items()}
        for brand, values in config["split_quotas_per_brand_and_label"].items()
    }
    sample = pd.read_csv(args.sample.resolve())
    required = {"source_id", "text", "rating", "brand", "weak_label", "normalized_text"}
    if missing := required - set(sample.columns):
        raise ValueError(f"В основной выборке отсутствуют поля: {sorted(missing)}")
    if len(sample) != 1000:
        raise ValueError(f"Ожидалось 1000 строк, найдено {len(sample)}")

    original = assign_brand_splits(sample, quotas, seed + 1)
    original["label"] = original["weak_label"].astype("int8")
    original["label_type"] = "weak_rating"
    original["source_type"] = "geo_reviews"
    original["group_id"] = original["normalized_text"].map(
        lambda value: stable_hash(value, 0)
    )
    split_order = {name: index for index, name in enumerate(SPLIT_ORDER)}
    brand_order = {brand: index for index, brand in enumerate(quotas)}
    original["_split_order"] = original["split"].map(split_order)
    original["_brand_order"] = original["brand"].map(brand_order)
    original = (
        original.sort_values(["_split_order", "_brand_order", "label", "source_id"])
        .drop(columns=["_split_order", "_brand_order"])
        .reset_index(drop=True)
    )

    manual_rows = prepare_manual(args.annotation_root.resolve())
    overlap = set(original["normalized_text"]) & set(manual_rows["normalized_text"])
    if overlap:
        raise ValueError(f"Geo Reviews пересекается с Otzyvru: {len(overlap)} текстов")
    combined = pd.concat([original, manual_rows], ignore_index=True, sort=False)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original.to_csv(output / "original.csv", index=False)
    combined.to_csv(output / "manual.csv", index=False)

    summary = {
        "random_seed": seed,
        "original_rows": len(original),
        "manual_rows": len(combined),
        "manual_rows_added": len(manual_rows),
        "original_split_counts": {
            str(split): int(count)
            for split, count in original["split"].value_counts().items()
        },
        "manual_split_counts": {
            str(split): int(count)
            for split, count in combined["split"].value_counts().items()
        },
        "original_split_class_counts": {
            str(split): {
                str(int(label)): int(count)
                for label, count in part["label"].value_counts().sort_index().items()
            }
            for split, part in original.groupby("split", sort=False)
        },
        "original_brand_label_split_counts": nested_counts(original, "split"),
        "manual_brand_counts": {
            str(brand): int(count)
            for brand, count in manual_rows["brand"].value_counts().items()
        },
        "manual_class_counts": {
            str(int(label)): int(count)
            for label, count in manual_rows["label"].value_counts().sort_index().items()
        },
        "geo_manual_normalized_overlap": len(overlap),
        "validation_digest_original": frame_digest(
            original.loc[original["split"].eq("validation")]
        ),
        "validation_digest_manual": frame_digest(
            combined.loc[combined["split"].eq("validation")]
        ),
        "test_digest_original": frame_digest(original.loc[original["split"].eq("test")]),
        "test_digest_manual": frame_digest(combined.loc[combined["split"].eq("test")]),
    }
    if summary["validation_digest_original"] != summary["validation_digest_manual"]:
        raise ValueError("Validation различается между Original и Manual")
    if summary["test_digest_original"] != summary["test_digest_manual"]:
        raise ValueError("Test различается между Original и Manual")
    (output / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
