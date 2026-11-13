"""Подготовка локальных рабочих выборок фастфуд-отзывов Geo Reviews.

Исходный Parquet не изменяется. Полные тексты и построчные результаты пишутся
только в ``data/processed``, который исключён из Git. В отслеживаемый каталог
``reports/figures`` попадают лишь агрегированные графики без текстов отзывов.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.fastfood.fastfood_data import normalize_text, stable_hash
from scripts.geo_reviews.validate_geo_reviews import (
    DEFAULT_DATA_ROOT,
    EXPECTED_ROWS,
    parquet_path,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "geo_fastfood"
DEFAULT_FIGURES = ROOT / "reports" / "figures"
SEED = 20261118
SAMPLE_SIZES = (200, 500, 1000)
FASTFOOD_RUBRIC = "Быстрое питание"

# Эти рубрики не доказывают ошибку, но делают предметную границу неоднозначной.
AMBIGUOUS_RUBRICS = {
    "Бар, паб",
    "Банкетный зал",
    "Гостиница",
    "Ресторан",
    "Столовая",
}

NETWORK_PATTERNS = (
    ("KFC", r"\bkfc\b|\bкфс\b"),
    ("Burger King", r"бургер\s*кинг|burger\s*king"),
    ("Dodo Pizza", r"додо\s*пицц|dodo\s*pizza"),
    ("Teremok", r"^теремок(?:\b|\s)"),
    ("Subway", r"\bsubway\b|\bсабв[еэ]й\b"),
    ("Rostic's", r"ростик|rostic"),
    ("Vkusno i tochka", r"вкусно\s*[-–—]?\s*и\s*[-–—]?\s*точка"),
    ("McDonald's", r"макдонал|mcdonald"),
    ("Kroshka Kartoshka", r"крошка\s*картош"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def rubric_set(value: object) -> set[str]:
    return {part.strip() for part in str(value).split(";") if part.strip()}


def serialize_rubric_set(values: set[str]) -> str:
    """Сериализовать множество рубрик в стабильном порядке для CSV."""
    return ";".join(sorted(values))


def csv_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Подготовить копию таблицы к CSV-экспорту, не меняя рабочие типы."""
    exported = frame.copy()
    if "rubric_set" in exported:
        exported["rubric_set"] = exported["rubric_set"].map(serialize_rubric_set)
    return exported


def network_name(names: pd.Series) -> pd.Series:
    result = pd.Series("Other / independent", index=names.index, dtype="string")
    source = names.fillna("").astype(str)
    for label, pattern in reversed(NETWORK_PATTERNS):
        result.loc[source.str.contains(pattern, case=False, regex=True)] = label
    return result


def balanced_sample(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size % 2:
        raise ValueError("Размер сбалансированной выборки должен быть чётным")
    per_class = size // 2
    groups = []
    for label in (0, 1):
        subset = frame.loc[frame["weak_label"].eq(label)].copy()
        if len(subset) < per_class:
            raise ValueError(
                f"Для класса {label} доступно {len(subset)}, требуется {per_class}"
            )
        subset["_sample_key"] = subset["source_id"].map(
            lambda value: stable_hash(value, seed + size)
        )
        groups.append(subset.sort_values("_sample_key").head(per_class))
    sample = pd.concat(groups, ignore_index=True).sort_values("_sample_key")
    return sample.drop(columns="_sample_key").reset_index(drop=True)


def save_figures(eligible: pd.DataFrame, summary: dict, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    eligible["weak_label_name"].value_counts().reindex(
        ["negative", "positive"], fill_value=0
    ).plot.bar(ax=axes[0], color=["#E15759", "#59A14F"])
    axes[0].set(title="Слабые классы фастфуд-пула", xlabel="", ylabel="Отзывы")
    axes[0].tick_params(axis="x", rotation=0)
    eligible["rating"].value_counts().sort_index().plot.bar(
        ax=axes[1], color="#4C78A8"
    )
    axes[1].set(title="Рейтинги фастфуд-пула", xlabel="Рейтинг", ylabel="Отзывы")
    axes[1].tick_params(axis="x", rotation=0)
    axes[2].hist(eligible["char_len"].clip(upper=1500), bins=50, color="#F28E2B")
    axes[2].set(
        title="Длины текстов (1500+ объединены)",
        xlabel="Символы",
        ylabel="Отзывы",
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "geo_fastfood_overview.png", dpi=160)
    plt.close(fig)

    labels = {
        "excluded": "Вне правил отбора",
        "explicit": "Явно подходящие",
        "review": "Требуют проверки",
    }
    order = ["excluded", "explicit", "review"]
    reasons = pd.Series(summary["selection_counts"]).reindex(order, fill_value=0)
    reason_labels = [labels[key] for key in order]
    colors = ["#BAB0AC", "#59A14F", "#F28E2B"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    reasons.plot.bar(ax=axes[0], color=colors)
    axes[0].set(
        title="Все 500 000 отзывов (единый масштаб)",
        xlabel="",
        ylabel="Отзывы",
    )
    axes[0].set_xticklabels(reason_labels, rotation=15, ha="right")
    axes[0].grid(axis="y", alpha=0.2)
    for index, value in enumerate(reasons):
        axes[0].text(
            index,
            value,
            f"{value:,}".replace(",", " "),
            ha="center",
            va="bottom",
        )

    detail = reasons.loc[["explicit", "review"]]
    detail.plot.bar(ax=axes[1], color=colors[1:])
    axes[1].set(
        title="Детализация отобранных категорий",
        xlabel="",
        ylabel="Отзывы (отдельный масштаб)",
    )
    axes[1].set_xticklabels(reason_labels[1:], rotation=15, ha="right")
    axes[1].grid(axis="y", alpha=0.2)
    for index, value in enumerate(detail):
        axes[1].text(
            index,
            value,
            f"{value:,}".replace(",", " "),
            ha="center",
            va="bottom",
        )
    fig.suptitle("Результат предметного отбора Geo Reviews")
    fig.tight_layout()
    fig.savefig(figures_dir / "geo_fastfood_selection.png", dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    source = parquet_path(args.data_root)
    if not source.is_file():
        raise FileNotFoundError(f"Не найден Geo Reviews: {source}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(source, columns=["name_ru", "rubrics", "rating", "text"])
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Ожидалось {EXPECTED_ROWS} строк Geo Reviews, найдено {len(frame)}")
    frame.insert(0, "source_id", [f"geo:{idx}" for idx in frame.index])
    frame["rubric_set"] = frame["rubrics"].map(rubric_set)
    frame["has_fastfood_rubric"] = frame["rubric_set"].map(
        lambda values: FASTFOOD_RUBRIC in values
    )
    frame["ambiguous_rubrics"] = frame["rubric_set"].map(
        lambda values: ";".join(sorted(values & AMBIGUOUS_RUBRICS))
    )
    frame["network"] = network_name(frame["name_ru"])
    known_name = frame["network"].ne("Other / independent")

    explicit = frame["has_fastfood_rubric"] & (
        frame["ambiguous_rubrics"].eq("") | known_name
    )
    frame["selection_status"] = np.select(
        [
            explicit,
            frame["has_fastfood_rubric"],
            known_name,
        ],
        ["explicit", "review", "review"],
        default="excluded",
    )
    frame["selection_reason"] = np.select(
        [
            explicit & known_name & frame["ambiguous_rubrics"].ne(""),
            explicit,
            frame["has_fastfood_rubric"] & frame["ambiguous_rubrics"].ne(""),
            known_name,
        ],
        [
            "fastfood_rubric_and_known_chain_name",
            "fastfood_rubric_without_ambiguous_co-rubric",
            "fastfood_rubric_with_ambiguous_co-rubric",
            "known_chain_name_without_fastfood_rubric",
        ],
        default="outside_rules",
    )
    frame["weak_label"] = np.select(
        [frame["rating"].isin([1, 2]), frame["rating"].isin([4, 5])],
        [0, 1],
        default=-1,
    ).astype("int8")
    frame["weak_label_name"] = frame["weak_label"].map(
        {-1: "excluded", 0: "negative", 1: "positive"}
    )
    frame["normalized_text"] = frame["text"].map(normalize_text)
    frame["char_len"] = frame["text"].str.len()
    frame["word_len"] = frame["text"].str.count(r"\S+")

    candidates = frame.loc[frame["selection_status"].ne("excluded")].copy()
    candidates.drop(columns="rubric_set").to_parquet(output / "all_candidates.parquet", index=False)
    review = candidates.loc[candidates["selection_status"].eq("review")]
    csv_export_frame(review).to_csv(output / "review_candidates.csv", index=False)

    eligible_raw = candidates.loc[
        candidates["selection_status"].eq("explicit")
        & candidates["weak_label"].isin([0, 1])
        & candidates["normalized_text"].ne("")
    ].copy()
    group_label_counts = eligible_raw.groupby("normalized_text")["weak_label"].nunique()
    conflicts = set(group_label_counts[group_label_counts.gt(1)].index)
    eligible_no_conflicts = eligible_raw.loc[
        ~eligible_raw["normalized_text"].isin(conflicts)
    ]
    eligible = eligible_no_conflicts.drop_duplicates("normalized_text", keep="first").copy()
    eligible.drop(columns="rubric_set", errors="ignore").to_parquet(
        output / "eligible_unique.parquet", index=False
    )

    produced_sizes: list[int] = []
    for size in SAMPLE_SIZES:
        try:
            sample = balanced_sample(eligible, size, args.seed)
        except ValueError as error:
            print(f"[WARNING] {error}")
            continue
        csv_export_frame(sample).to_csv(output / f"sample_{size:04d}.csv", index=False)
        produced_sizes.append(size)

    max_balanced = 2 * int(eligible["weak_label"].value_counts().min())
    max_sample = balanced_sample(eligible, max_balanced, args.seed)
    max_sample.to_parquet(output / "sample_max_balanced.parquet", index=False)

    selection_counts = Counter(frame["selection_status"])
    summary = {
        "source": str(source),
        "seed": args.seed,
        "source_rows": len(frame),
        "selection_counts": dict(sorted(selection_counts.items())),
        "selection_reasons": frame["selection_reason"].value_counts().to_dict(),
        "candidate_rows": len(candidates),
        "explicit_rows": int(candidates["selection_status"].eq("explicit").sum()),
        "review_rows": int(candidates["selection_status"].eq("review").sum()),
        "explicit_weak_label_rows_before_dedup": len(eligible_raw),
        "normalized_duplicate_extra": len(eligible_no_conflicts) - len(eligible),
        "normalized_conflict_groups": len(conflicts),
        "eligible_unique_rows": len(eligible),
        "eligible_class_counts": {
            str(key): int(value)
            for key, value in eligible["weak_label_name"].value_counts().items()
        },
        "rating_counts": {
            str(key): int(value)
            for key, value in eligible["rating"].value_counts().sort_index().items()
        },
        "length_chars": {
            "min": int(eligible["char_len"].min()),
            "median": float(eligible["char_len"].median()),
            "mean": float(eligible["char_len"].mean()),
            "p95": float(eligible["char_len"].quantile(0.95)),
            "max": int(eligible["char_len"].max()),
        },
        "top_networks": {
            str(key): int(value)
            for key, value in eligible["network"].value_counts().head(20).items()
        },
        "sample_sizes": produced_sizes,
        "max_balanced_size": max_balanced,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_figures(eligible, summary, args.figures_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
