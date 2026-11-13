"""Проверить заполненные результаты локального контроля Geo Reviews.

Скрипт ничего не исправляет и не переносит ручные решения в рабочий датасет.
Агрегированная сводка без текстов сохраняется только в ``.local``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = ROOT / ".local" / "geo_quality_checks"
QUALITY_ORIGINAL = "weak_label_manual_check_50.csv"
QUALITY_FILLED = "weak_label_manual_check_50_filled.csv"
DOMAIN_ORIGINAL = "other_independent_review_30.csv"
DOMAIN_FILLED = "other_independent_review_30_final.csv"
EXPECTED_AMBIGUOUS_ID = "geo:362321"
EXPECTED_UNRESOLVED_DOMAIN_IDS = {
    "geo:438599",
    "geo:19620",
    "geo:114034",
    "geo:182237",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False, dtype={"source_id": str})


def assert_unchanged_except_manual(
    original: pd.DataFrame,
    filled: pd.DataFrame,
    manual_column: str,
) -> None:
    if list(original.columns) != list(filled.columns):
        raise ValueError(f"Наборы столбцов различаются для {manual_column}")
    if original["source_id"].duplicated().any() or filled["source_id"].duplicated().any():
        raise ValueError(f"Повторяющиеся ID в файле {manual_column}")
    if set(original["source_id"]) != set(filled["source_id"]):
        raise ValueError(f"Наборы ID различаются для {manual_column}")
    stable_columns = [column for column in original.columns if column != manual_column]
    left = original.sort_values("source_id")[stable_columns].reset_index(drop=True)
    right = filled.sort_values("source_id")[stable_columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def main() -> int:
    args = parse_args()
    root = args.quality_root.expanduser().resolve()
    paths = {
        name: root / name
        for name in (QUALITY_ORIGINAL, QUALITY_FILLED, DOMAIN_ORIGINAL, DOMAIN_FILLED)
    }
    quality_original = read_csv(paths[QUALITY_ORIGINAL])
    quality = read_csv(paths[QUALITY_FILLED])
    domain_original = read_csv(paths[DOMAIN_ORIGINAL])
    domain = read_csv(paths[DOMAIN_FILLED])
    assert_unchanged_except_manual(
        quality_original, quality, "manual_quality_check"
    )
    assert_unchanged_except_manual(
        domain_original, domain, "manual_fastfood_check"
    )

    if len(quality) != 50 or len(domain) != 30:
        raise ValueError("Ожидались контрольные выборки на 50 и 30 строк")
    quality_values = set(quality["manual_quality_check"])
    if not quality_values.issubset({"0", "1", "ambiguous"}):
        raise ValueError(f"Неожиданные решения тональности: {quality_values}")
    ambiguous_ids = set(
        quality.loc[quality["manual_quality_check"].eq("ambiguous"), "source_id"]
    )
    if ambiguous_ids != {EXPECTED_AMBIGUOUS_ID}:
        raise ValueError(f"Неожиданные неоднозначные ID: {sorted(ambiguous_ids)}")
    determined = quality.loc[quality["manual_quality_check"].isin(["0", "1"])].copy()
    determined["manual_label"] = determined["manual_quality_check"].astype(int)
    matches = int(determined["manual_label"].eq(determined["weak_label"]).sum())
    mismatches = len(determined) - matches

    domain_values = set(domain["manual_fastfood_check"])
    if not domain_values.issubset({"", "0", "1"}):
        raise ValueError(f"Неожиданные решения предметной проверки: {domain_values}")
    unresolved_ids = set(
        domain.loc[domain["manual_fastfood_check"].eq(""), "source_id"]
    )
    if unresolved_ids != EXPECTED_UNRESOLVED_DOMAIN_IDS:
        raise ValueError(f"Неожиданные неопределённые ID: {sorted(unresolved_ids)}")
    domain_counts = domain["manual_fastfood_check"].value_counts().to_dict()
    confirmed = int(domain_counts.get("1", 0))
    rejected = int(domain_counts.get("0", 0))
    defined = confirmed + rejected

    summary = {
        "status": "PASSED",
        "quality_check": {
            "rows": len(quality),
            "determined": len(determined),
            "matches": matches,
            "mismatches": mismatches,
            "ambiguous": len(ambiguous_ids),
            "ambiguous_ids": sorted(ambiguous_ids),
            "interpretation": "49/49 определённых решений, не оценка точности всего Geo Reviews",
        },
        "fastfood_domain_check": {
            "rows": len(domain),
            "confirmed": confirmed,
            "rejected": rejected,
            "unresolved": len(unresolved_ids),
            "unresolved_ids": sorted(unresolved_ids),
            "confirmed_among_defined": confirmed / defined,
            "interpretation": "контроль 30 строк, не оценка точности всего фильтра",
        },
        "cross_sample_id_overlap": sorted(
            set(quality["source_id"]) & set(domain["source_id"])
        ),
        "source_files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    if matches != 49 or mismatches != 0 or confirmed != 24 or rejected != 2:
        raise ValueError(f"Контрольные итоги не совпали с ожидаемыми: {summary}")
    output = root / "results_summary.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
