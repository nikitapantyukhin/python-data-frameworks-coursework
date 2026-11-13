"""Построить агрегированный EDA первого датасета без публикации текстов."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.fastfood.quality.audit_fastfood_annotations import (
    canonical,
    find_column,
    locate_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GEO = ROOT / "data" / "processed" / "geo_fastfood" / "sample_1000.csv"
DEFAULT_ANNOTATIONS = ROOT / ".local" / "fastfood_annotation"
DEFAULT_FIGURES = ROOT / "reports" / "figures"
DEFAULT_SUMMARY = ROOT / "data" / "processed" / "fastfood_eda_summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geo-sample", type=Path, default=DEFAULT_GEO)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Путь для JSON-сводки EDA",
    )
    return parser.parse_args(argv)


def describe_lengths(values: pd.Series) -> dict[str, float]:
    lengths = values.fillna("").astype(str).str.len()
    return {
        "min": int(lengths.min()),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "p95": float(lengths.quantile(0.95)),
        "max": int(lengths.max()),
    }


def main() -> int:
    args = parse_args()
    figures = args.figures_dir.resolve()
    figures.mkdir(parents=True, exist_ok=True)
    geo = pd.read_csv(args.geo_sample)
    summary: dict = {
        "geo_rows": len(geo),
        "geo_class_counts": {
            str(int(key)): int(value)
            for key, value in geo["weak_label"].value_counts().sort_index().items()
        },
        "geo_rating_counts": {
            str(key): int(value)
            for key, value in geo["rating"].value_counts().sort_index().items()
        },
        "geo_length_chars": describe_lengths(geo["text"]),
        "manual_status": "not_available",
    }
    try:
        bundle = locate_bundle(args.annotation_root)
    except FileNotFoundError:
        bundle = None
    if bundle is not None:
        raw_final = pd.read_csv(bundle / "04_final_dataset.csv")
        manual = canonical(raw_final)
        provenance = pd.read_csv(bundle / "05_provenance.csv")
        network_col = find_column(provenance, "network", required=False)
        rating_col = find_column(provenance, "rating", required=False)
        provenance_id = find_column(provenance, "id", required=False)
        if network_col is not None and provenance_id is not None:
            networks = pd.DataFrame(
                {
                    "id": provenance[provenance_id].astype(str).str.strip(),
                    "network": provenance[network_col].fillna("Не указано").astype(str),
                    "source_rating": (
                        provenance[rating_col]
                        if rating_col is not None
                        else pd.Series(pd.NA, index=provenance.index)
                    ),
                }
            )
            manual = manual.merge(networks, on="id", how="left", validate="one_to_one")
            manual["network"] = manual["network"].fillna("Не указано")
        elif network_col is not None and len(provenance) == len(manual):
            manual["network"] = provenance[network_col].fillna("Не указано").astype(str).to_numpy()
        else:
            manual["network"] = "Не указано"
        if "source_rating" not in manual:
            manual["source_rating"] = pd.NA
        summary.update(
            {
                "manual_status": "available",
                "manual_rows": len(manual),
                "manual_class_counts": {
                    str(int(key)): int(value)
                    for key, value in manual["label"].value_counts().sort_index().items()
                },
                "manual_network_counts": {
                    str(key): int(value)
                    for key, value in manual["network"].value_counts().items()
                },
                "manual_network_class_counts": {
                    str(network): {
                        str(int(label)): int(count)
                        for label, count in group["label"].value_counts().sort_index().items()
                    }
                    for network, group in manual.groupby("network", sort=True)
                },
                "manual_source_rating_counts": {
                    str(int(key)): int(value)
                    for key, value in manual["source_rating"]
                    .dropna()
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "manual_length_chars": describe_lengths(manual["text"]),
            }
        )
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        manual["label"].map({0: "negative", 1: "positive"}).value_counts().plot.bar(
            ax=axes[0], color=["#59A14F", "#E15759"]
        )
        axes[0].set(title="Классы Otzyvru", xlabel="", ylabel="Отзывы")
        manual["network"].value_counts().plot.bar(ax=axes[1], color="#4C78A8")
        axes[1].set(title="Сети Otzyvru", xlabel="", ylabel="Отзывы")
        axes[2].hist(
            [geo["text"].str.len(), manual["text"].str.len()],
            bins=40,
            density=True,
            alpha=0.6,
            label=["Geo Reviews", "Otzyvru"],
        )
        axes[2].set(title="Сравнение длин", xlabel="Символы", ylabel="Плотность")
        axes[2].legend()
        for axis in axes:
            axis.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures / "fastfood_manual_comparison.png", dpi=160)
        plt.close(fig)
    output = args.summary_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
