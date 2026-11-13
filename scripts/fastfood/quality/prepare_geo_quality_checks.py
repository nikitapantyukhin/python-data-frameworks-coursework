"""Подготовить локальные файлы для ручного контроля рабочей Geo-выборки.

Скрипт не оценивает тональность, не меняет слабые метки и не изменяет
train/validation/test. Файлы с текстами создаются только внутри ``.local``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.fastfood.fastfood_data import stable_hash


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "data" / "processed" / "fastfood_experiment" / "original.csv"
DEFAULT_OUTPUT = ROOT / ".local" / "geo_quality_checks"
SEED = 20261118

QUALITY_QUOTAS = {
    "train": {0: 17, 1: 17},
    "validation": {0: 4, 1: 4},
    "test": {0: 4, 1: 4},
}
OTHER_QUOTAS = {
    "train": {0: 10, 1: 10},
    "validation": {0: 3, 1: 3},
    "test": {0: 2, 1: 2},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def stratified_sample(
    frame: pd.DataFrame,
    quotas: dict[str, dict[int, int]],
    seed: int,
) -> pd.DataFrame:
    """Детерминированно отобрать заданное число строк каждой части и класса."""

    selected: list[pd.DataFrame] = []
    for split, class_quotas in quotas.items():
        for label, count in class_quotas.items():
            subset = frame.loc[
                frame["split"].eq(split) & frame["label"].eq(label)
            ].copy()
            if len(subset) < count:
                raise ValueError(
                    f"Недостаточно строк для split={split}, label={label}: "
                    f"{len(subset)} < {count}"
                )
            subset["_review_key"] = subset["source_id"].map(
                lambda value: stable_hash(value, seed)
            )
            selected.append(subset.sort_values("_review_key").head(count))
    return (
        pd.concat(selected, ignore_index=True)
        .sort_values(["split", "label", "_review_key"])
        .drop(columns="_review_key")
        .reset_index(drop=True)
    )


def counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    table = pd.crosstab(frame["split"], frame["label"])
    return {
        str(split): {str(int(label)): int(value) for label, value in row.items()}
        for split, row in table.iterrows()
    }


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    frame = pd.read_csv(source)
    required = {
        "source_id",
        "name_ru",
        "rubrics",
        "rating",
        "text",
        "label",
        "split",
        "network",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"В исходной выборке отсутствуют поля: {sorted(missing)}")
    if len(frame) != 1000:
        raise ValueError(f"Ожидалось 1 000 строк, найдено {len(frame)}")
    if set(frame["label"]) != {0, 1}:
        raise ValueError("Слабые метки должны содержать оба класса 0 и 1")
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise ValueError("Ожидались части train, validation и test")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    quality = stratified_sample(frame, QUALITY_QUOTAS, args.seed + 50)
    quality = quality[
        ["source_id", "text", "rating", "label", "split"]
    ].rename(columns={"label": "weak_label"})
    quality["manual_quality_check"] = ""
    quality_path = output / "weak_label_manual_check_50.csv"
    quality.to_csv(quality_path, index=False)

    other_source = frame.loc[frame["network"].eq("Other / independent")].copy()
    other = stratified_sample(other_source, OTHER_QUOTAS, args.seed + 30)
    other = other[
        [
            "source_id",
            "name_ru",
            "rubrics",
            "text",
            "rating",
            "label",
            "split",
            "network",
        ]
    ].rename(columns={"label": "weak_label"})
    other["manual_fastfood_check"] = ""
    other_path = output / "other_independent_review_30.csv"
    other.to_csv(other_path, index=False)

    manifest = {
        "source": str(source),
        "seed": args.seed,
        "weak_label_manual_check": {
            "path": str(quality_path),
            "rows": len(quality),
            "counts": counts(quality.rename(columns={"weak_label": "label"})),
            "manual_column": "manual_quality_check",
            "manual_column_filled": False,
        },
        "other_independent_review": {
            "path": str(other_path),
            "rows": len(other),
            "counts": counts(other.rename(columns={"weak_label": "label"})),
            "manual_column": "manual_fastfood_check",
            "manual_column_filled": False,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
