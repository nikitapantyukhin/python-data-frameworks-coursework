"""Построить агрегированный EDA актуальной четырёхбрендовой выборки."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE = ROOT / "data" / "processed" / "geo_fastfood_brands" / "sample_1000.csv"
DEFAULT_ORIGINAL = ROOT / "data" / "processed" / "fastfood_experiment" / "original.csv"
DEFAULT_MANUAL = ROOT / "data" / "processed" / "fastfood_experiment" / "manual.csv"
DEFAULT_PROVENANCE = ROOT / ".local" / "fastfood_annotation" / "05_provenance.csv"
DEFAULT_FIGURES = ROOT / "reports" / "figures"
DEFAULT_SUMMARY = ROOT / "data" / "processed" / "geo_fastfood_brands" / "eda_summary.json"
BRAND_ORDER = ["KFC", "Burger King", "Вкусно — и точка", "Rostic's"]
SPLIT_ORDER = ["train", "validation", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--manual", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def save_brand_distribution(sample: pd.DataFrame, figures: Path) -> None:
    counts = sample["brand"].value_counts().reindex(BRAND_ORDER)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(counts.index, counts.values, color="#4C78A8")
    ax.bar_label(bars)
    ax.set(title="Основная Geo-выборка по брендам", ylabel="Отзывы")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "geo_brands_distribution.png", dpi=160)
    plt.close(fig)


def save_brand_sentiment(sample: pd.DataFrame, figures: Path) -> None:
    table = pd.crosstab(sample["brand"], sample["weak_label"]).reindex(BRAND_ORDER)
    table = table.rename(columns={0: "negative", 1: "positive"})
    ax = table.plot.bar(figsize=(9, 4.5), color=["#E15759", "#59A14F"])
    ax.set(title="Баланс классов внутри брендов", xlabel="", ylabel="Отзывы")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    ax.figure.tight_layout()
    ax.figure.savefig(figures / "geo_brands_sentiment.png", dpi=160)
    plt.close(ax.figure)


def save_ratings(sample: pd.DataFrame, figures: Path) -> None:
    table = pd.crosstab(sample["brand"], sample["rating"]).reindex(BRAND_ORDER)
    table = table.reindex(columns=[1, 2, 4, 5], fill_value=0)
    ax = table.plot.bar(figsize=(10, 5), color=["#D62728", "#FF9896", "#98DF8A", "#2CA02C"])
    ax.set(title="Рейтинги в основной выборке", xlabel="", ylabel="Отзывы")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    ax.figure.tight_layout()
    ax.figure.savefig(figures / "geo_brands_ratings.png", dpi=160)
    plt.close(ax.figure)


def save_lengths(sample: pd.DataFrame, figures: Path) -> None:
    values = [sample.loc[sample["brand"].eq(brand), "char_len"] for brand in BRAND_ORDER]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(values, tick_labels=BRAND_ORDER, showfliers=False)
    ax.set(title="Длины текстов по брендам (без выбросов)", ylabel="Символы")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "geo_brands_text_lengths.png", dpi=160)
    plt.close(fig)


def save_splits(original: pd.DataFrame, figures: Path) -> None:
    table = pd.crosstab(original["brand"], original["split"]).reindex(BRAND_ORDER)
    table = table.reindex(columns=SPLIT_ORDER)
    ax = table.plot.bar(stacked=True, figsize=(9, 4.5), color=["#4C78A8", "#F28E2B", "#B279A2"])
    ax.set(title="Распределение брендов по частям", xlabel="", ylabel="Отзывы")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    ax.figure.tight_layout()
    ax.figure.savefig(figures / "geo_brands_splits.png", dpi=160)
    plt.close(ax.figure)


def save_source_comparison(sample: pd.DataFrame, provenance: pd.DataFrame, figures: Path) -> None:
    manual_names = provenance["restaurant"].replace({"McDonald’s": "McDonald's"})
    order = ["KFC", "Burger King", "Вкусно — и точка", "Rostic's", "McDonald's"]
    table = pd.DataFrame(
        {
            "Geo Reviews": sample["brand"].value_counts().reindex(order, fill_value=0),
            "Otzyvru": manual_names.value_counts().reindex(order, fill_value=0),
        }
    )
    ax = table.plot.bar(figsize=(10, 4.5), color=["#4C78A8", "#F28E2B"])
    ax.set(title="Сравнение брендов Geo Reviews и Otzyvru", xlabel="", ylabel="Отзывы")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    ax.figure.tight_layout()
    ax.figure.savefig(figures / "geo_vs_otzyvru_brands.png", dpi=160)
    plt.close(ax.figure)


def save_regime_sizes(original: pd.DataFrame, manual: pd.DataFrame, figures: Path) -> None:
    table = pd.DataFrame(
        {
            "Original": original["split"].value_counts().reindex(SPLIT_ORDER),
            "Manual": manual["split"].value_counts().reindex(SPLIT_ORDER),
        }
    )
    ax = table.plot.bar(figsize=(8, 4.5), color=["#4C78A8", "#59A14F"])
    ax.set(title="Размеры режимов до и после ручного обогащения", xlabel="", ylabel="Отзывы")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.2)
    ax.figure.tight_layout()
    ax.figure.savefig(figures / "original_vs_manual_sizes.png", dpi=160)
    plt.close(ax.figure)


def main() -> int:
    args = parse_args()
    sample = pd.read_csv(args.sample.resolve())
    original = pd.read_csv(args.original.resolve())
    manual = pd.read_csv(args.manual.resolve())
    provenance = pd.read_csv(args.provenance.resolve())
    figures = args.figures_dir.resolve()
    figures.mkdir(parents=True, exist_ok=True)

    save_brand_distribution(sample, figures)
    save_brand_sentiment(sample, figures)
    save_ratings(sample, figures)
    save_lengths(sample, figures)
    save_splits(original, figures)
    save_source_comparison(sample, provenance, figures)
    save_regime_sizes(original, manual, figures)

    lengths = sample.groupby("brand", sort=False)["char_len"].agg(
        ["min", "median", "mean", "max"]
    )
    summary = {
        "sample_rows": len(sample),
        "brand_counts": {
            str(brand): int(count) for brand, count in sample["brand"].value_counts().items()
        },
        "brand_class_counts": {
            str(brand): {
                str(int(label)): int(count)
                for label, count in part["weak_label"].value_counts().sort_index().items()
            }
            for brand, part in sample.groupby("brand", sort=False)
        },
        "rating_counts": {
            str(int(rating)): int(count)
            for rating, count in sample["rating"].value_counts().sort_index().items()
        },
        "length_chars_by_brand": {
            str(brand): {
                key: (int(value) if key in {"min", "max"} else float(value))
                for key, value in row.items()
            }
            for brand, row in lengths.iterrows()
        },
        "original_split_counts": {
            str(split): int(count) for split, count in original["split"].value_counts().items()
        },
        "manual_split_counts": {
            str(split): int(count) for split, count in manual["split"].value_counts().items()
        },
        "otzyvru_brand_counts": {
            str(brand): int(count)
            for brand, count in provenance["restaurant"].replace({"McDonald’s": "McDonald's"}).value_counts().items()
        },
    }
    output = args.summary_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
