"""Сформировать основную Geo Reviews выборку по четырём сетям.

Скрипт не изменяет исходный Parquet и файлы ручной разметки. Полные тексты
сохраняются только в Git-игнорируемом ``data/processed``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.fastfood.fastfood_data import (
    normalize_text,
    rubric_set,
    stable_hash,
    weak_label_from_rating,
)
from scripts.geo_reviews.validate_geo_reviews import DEFAULT_DATA_ROOT, parquet_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "fastfood_experiment.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "geo_fastfood_brands"
DEFAULT_ANNOTATIONS = ROOT / ".local" / "fastfood_annotation"
FASTFOOD_RUBRIC = "Быстрое питание"
BRAND_MATCH_RULE = "normalized_name_whitelist+exact_fastfood_rubric"

BRAND_NAME_WHITELIST: dict[str, tuple[str, ...]] = {
    "KFC": ("KFC", "KFC Авто", "KFC shell Ржавки"),
    "Burger King": ("Бургер Кинг",),
    "Вкусно — и точка": (
        "Вкусно — и точка",
        "Вкусно — и точка. Авто",
        "Вкусно — и точка. Киоск",
    ),
    "Rostic's": ("Rostic's", "Rostic's Авто"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_brand_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for brand, names in BRAND_NAME_WHITELIST.items():
        for name in names:
            key = normalize_text(name)
            previous = lookup.get(key)
            if previous is not None and previous != brand:
                raise ValueError(f"Название {name!r} сопоставлено нескольким брендам")
            lookup[key] = brand
    return lookup


def match_brand(name: object, rubrics: object) -> tuple[str | None, str | None]:
    """Сопоставить организацию только по whitelist и точной рубрике."""

    if FASTFOOD_RUBRIC not in rubric_set(rubrics):
        return None, None
    brand = normalized_brand_lookup().get(normalize_text(name))
    return (brand, BRAND_MATCH_RULE) if brand is not None else (None, None)


def build_eligible_pool(
    frame: pd.DataFrame,
    manual_texts: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Построить очищенный уникальный пул и агрегаты отбора."""

    required = {"address", "name_ru", "rubrics", "rating", "text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"В Geo Reviews отсутствуют поля: {sorted(missing)}")

    work = frame.copy()
    work["_source_row"] = work.index.astype(int)
    work["source_id"] = "geo:" + work["_source_row"].astype(str)
    work["normalized_name"] = work["name_ru"].map(normalize_text)
    lookup = normalized_brand_lookup()
    work["brand"] = work["normalized_name"].map(lookup)
    work["has_fastfood_rubric"] = work["rubrics"].map(
        lambda value: FASTFOOD_RUBRIC in rubric_set(value)
    )
    work.loc[~work["has_fastfood_rubric"], "brand"] = pd.NA
    work["brand_match_rule"] = pd.NA
    work.loc[work["brand"].notna(), "brand_match_rule"] = BRAND_MATCH_RULE
    matched = work.loc[work["brand"].notna()].copy()

    matched["weak_label"] = matched["rating"].map(weak_label_from_rating).astype("int8")
    matched["normalized_text"] = matched["text"].map(normalize_text)
    matched["char_len"] = matched["text"].fillna("").astype(str).str.len()
    matched["word_len"] = matched["text"].fillna("").astype(str).str.count(r"\S+")

    usable = matched.loc[
        matched["weak_label"].isin([0, 1]) & matched["normalized_text"].ne("")
    ].copy()
    label_counts = usable.groupby("normalized_text")["weak_label"].nunique()
    brand_counts = usable.groupby("normalized_text")["brand"].nunique()
    label_conflicts = set(label_counts[label_counts.gt(1)].index)
    cross_brand_duplicates = set(brand_counts[brand_counts.gt(1)].index)
    excluded_groups = label_conflicts | cross_brand_duplicates
    without_conflicts = usable.loc[
        ~usable["normalized_text"].isin(excluded_groups)
    ].sort_values("_source_row")
    normalized_duplicate_extra = int(
        without_conflicts.duplicated("normalized_text").sum()
    )
    eligible = without_conflicts.drop_duplicates("normalized_text", keep="first").copy()

    manual_normalized = (
        set(manual_texts.fillna("").astype(str).map(normalize_text))
        if manual_texts is not None
        else set()
    )
    manual_normalized.discard("")
    manual_overlap = eligible["normalized_text"].isin(manual_normalized)
    manual_overlap_rows = int(manual_overlap.sum())
    eligible = eligible.loc[~manual_overlap].copy()

    name_counts = (
        matched.groupby(["brand", "name_ru"], dropna=False)
        .size()
        .sort_index()
    )
    summary: dict[str, Any] = {
        "source_rows": len(frame),
        "brand_match_rule": BRAND_MATCH_RULE,
        "matched_rows": len(matched),
        "matched_name_counts": {
            str(brand): {str(name): int(count) for (_, name), count in group.items()}
            for brand, group in name_counts.groupby(level=0)
        },
        "matched_rating_counts": {
            str(brand): {
                str(int(rating)): int(count)
                for rating, count in part["rating"].value_counts().sort_index().items()
            }
            for brand, part in matched.groupby("brand", sort=False)
        },
        "usable_rows_before_dedup": len(usable),
        "label_conflict_groups": len(label_conflicts),
        "cross_brand_duplicate_groups": len(cross_brand_duplicates),
        "normalized_duplicate_extra": normalized_duplicate_extra,
        "manual_normalized_overlap_rows_removed": manual_overlap_rows,
        "eligible_rows": len(eligible),
        "eligible_class_counts": {
            str(brand): {
                str(int(label)): int(count)
                for label, count in part["weak_label"].value_counts().sort_index().items()
            }
            for brand, part in eligible.groupby("brand", sort=False)
        },
        "eligible_rating_counts": {
            str(brand): {
                str(int(rating)): int(count)
                for rating, count in part["rating"].value_counts().sort_index().items()
            }
            for brand, part in eligible.groupby("brand", sort=False)
        },
    }
    return eligible.reset_index(drop=True), summary


def sample_by_quotas(
    pool: pd.DataFrame,
    quotas_per_label: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    """Детерминированно выполнить заданные бренд-классовые квоты."""

    groups: list[pd.DataFrame] = []
    for brand, quota in quotas_per_label.items():
        for label in (0, 1):
            subset = pool.loc[
                pool["brand"].eq(brand) & pool["weak_label"].eq(label)
            ].copy()
            if len(subset) < quota:
                raise ValueError(
                    f"Недостаточно отзывов {brand}, label={label}: "
                    f"доступно {len(subset)}, требуется {quota}"
                )
            subset["selection_key"] = subset["source_id"].map(
                lambda value: stable_hash(f"{brand}\0{label}\0{value}", seed)
            )
            groups.append(subset.sort_values("selection_key").head(quota))
    sample = pd.concat(groups, ignore_index=True)
    brand_order = {brand: index for index, brand in enumerate(quotas_per_label)}
    sample["_brand_order"] = sample["brand"].map(brand_order)
    return (
        sample.sort_values(["_brand_order", "weak_label", "selection_key"])
        .drop(columns=["_brand_order", "_source_row"], errors="ignore")
        .reset_index(drop=True)
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    seed = int(config["random_seed"])
    quotas = {
        str(brand): int(quota)
        for brand, quota in config["base_sample"]["sample_quotas_per_label"].items()
    }
    source = parquet_path(args.data_root)
    if not source.is_file():
        raise FileNotFoundError(source)
    manual_path = args.annotation_root.resolve() / "04_final_dataset.csv"
    if not manual_path.is_file():
        raise FileNotFoundError(manual_path)

    frame = pd.read_parquet(source, columns=["address", "name_ru", "rubrics", "rating", "text"])
    manual = pd.read_csv(manual_path)
    if "text" not in manual:
        raise ValueError("В 04_final_dataset.csv отсутствует поле text")
    pool, summary = build_eligible_pool(frame, manual["text"])
    sample = sample_by_quotas(pool, quotas, seed)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(output / "available_pool.parquet", index=False)
    sample.to_csv(output / "sample_1000.csv", index=False)
    summary.update(
        {
            "source": str(source),
            "random_seed": seed,
            "sample_rows": len(sample),
            "sample_quotas_per_label": quotas,
            "sample_class_counts": {
                str(int(label)): int(count)
                for label, count in sample["weak_label"].value_counts().sort_index().items()
            },
            "sample_brand_class_counts": {
                str(brand): {
                    str(int(label)): int(count)
                    for label, count in part["weak_label"].value_counts().sort_index().items()
                }
                for brand, part in sample.groupby("brand", sort=False)
            },
        }
    )
    (output / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
