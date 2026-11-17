"""Подготовить независимый ручной и fuzzy-контроль новой Geo-выборки.

Шаблоны не содержат подсказок о рейтинге или слабой метке. Если в существующих
файлах обнаружены ручные ответы, скрипт останавливается до любых изменений.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.fastfood.fastfood_data import normalize_text, stable_hash, token_bigrams
from scripts.fastfood.prepare_geo_brand_sample import DEFAULT_CONFIG, load_config


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORIGINAL = ROOT / "data" / "processed" / "fastfood_experiment" / "original.csv"
DEFAULT_MANUAL = ROOT / "data" / "processed" / "fastfood_experiment" / "manual.csv"
DEFAULT_POOL = ROOT / "data" / "processed" / "geo_fastfood_brands" / "available_pool.parquet"
DEFAULT_SAMPLE = ROOT / "data" / "processed" / "geo_fastfood_brands" / "sample_1000.csv"
DEFAULT_OUTPUT = ROOT / ".local" / "geo_brands_quality_checks"
PAIR_CATEGORIES = (
    "train_vs_validation",
    "train_vs_test",
    "validation_vs_test",
    "within_geo_train",
    "geo_vs_otzyvru",
    "within_otzyvru",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--manual", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_sentiment_reviews(
    original: pd.DataFrame,
    quotas_per_label: dict[str, int],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Выбрать слепую стратифицированную проверку только из train."""

    train = original.loc[original["split"].eq("train")].copy()
    selected: list[pd.DataFrame] = []
    for brand, quota in quotas_per_label.items():
        for label in (0, 1):
            part = train.loc[train["brand"].eq(brand) & train["weak_label"].eq(label)].copy()
            if len(part) < quota:
                raise ValueError(
                    f"Недостаточно train-строк {brand}, weak_label={label}: "
                    f"{len(part)} < {quota}"
                )
            part["_quality_key"] = part["source_id"].map(
                lambda value: stable_hash(f"sentiment\0{brand}\0{label}\0{value}", seed)
            )
            selected.append(part.sort_values("_quality_key").head(quota))
    result = pd.concat(selected, ignore_index=True)
    result["_blind_order"] = result["source_id"].map(
        lambda value: stable_hash(f"blind\0{value}", seed)
    )
    result = result.sort_values("_blind_order").reset_index(drop=True)
    annotation = result[["source_id", "text"]].copy()
    annotation["manual_sentiment"] = ""
    annotation["manual_notes"] = ""
    reference = result[["source_id", "brand", "rating", "weak_label", "split"]].copy()
    return annotation, reference


def build_brand_names_review(pool: pd.DataFrame) -> pd.DataFrame:
    """Собрать все включённые варианты названий без текстов отзывов."""

    grouped = (
        pool.groupby(["name_ru", "brand", "brand_match_rule"], dropna=False)
        .agg(
            rubrics=("rubrics", lambda values: " || ".join(sorted(set(map(str, values))))),
            review_count=("source_id", "size"),
        )
        .reset_index()
        .rename(columns={"brand": "brand_assigned"})
        .sort_values(["brand_assigned", "name_ru"])
        .reset_index(drop=True)
    )
    grouped["manual_brand_check"] = ""
    grouped["manual_notes"] = ""
    return grouped[
        [
            "name_ru",
            "brand_assigned",
            "rubrics",
            "brand_match_rule",
            "review_count",
            "manual_brand_check",
            "manual_notes",
        ]
    ]


def select_brand_records(
    sample: pd.DataFrame,
    brand_quotas: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    """Выбрать записи, максимально покрывая варианты ``name_ru``."""

    selected: list[pd.DataFrame] = []
    for brand, quota in brand_quotas.items():
        part = sample.loc[sample["brand"].eq(brand)].copy()
        if len(part) < quota:
            raise ValueError(f"Недостаточно строк {brand}: {len(part)} < {quota}")
        part["_record_key"] = part["source_id"].map(
            lambda value: stable_hash(f"brand_record\0{brand}\0{value}", seed)
        )
        representatives = (
            part.sort_values("_record_key").drop_duplicates("name_ru", keep="first")
        )
        if len(representatives) > quota:
            representatives = representatives.sort_values("_record_key").head(quota)
        remaining_count = quota - len(representatives)
        remaining = part.loc[~part["source_id"].isin(representatives["source_id"])]
        chosen = pd.concat(
            [representatives, remaining.sort_values("_record_key").head(remaining_count)],
            ignore_index=True,
        )
        selected.append(chosen)
    result = pd.concat(selected, ignore_index=True)
    result["_blind_order"] = result["source_id"].map(
        lambda value: stable_hash(f"brand_blind\0{value}", seed)
    )
    result = result.sort_values("_blind_order").rename(columns={"brand": "brand_assigned"})
    output = result[
        ["source_id", "name_ru", "rubrics", "address", "brand_assigned"]
    ].reset_index(drop=True)
    output["manual_brand_check"] = ""
    output["manual_notes"] = ""
    return output


def pair_category(left: pd.Series, right: pd.Series) -> str | None:
    left_source, right_source = left["source"], right["source"]
    if left_source == "geo_reviews" and right_source == "geo_reviews":
        splits = {left["split"], right["split"]}
        if splits == {"train", "validation"}:
            return "train_vs_validation"
        if splits == {"train", "test"}:
            return "train_vs_test"
        if splits == {"validation", "test"}:
            return "validation_vs_test"
        if left["split"] == right["split"] == "train":
            return "within_geo_train"
        return None
    if {left_source, right_source} == {"geo_reviews", "otzyvru"}:
        return "geo_vs_otzyvru"
    if left_source == right_source == "otzyvru":
        return "within_otzyvru"
    return None


def fuzzy_records(original: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    geo = original[["source_id", "text", "split"]].copy()
    geo["source"] = "geo_reviews"
    extra = manual.loc[manual["source_type"].eq("otzyvru"), ["source_id", "text", "split"]].copy()
    extra["source"] = "otzyvru"
    records = pd.concat([geo, extra], ignore_index=True)
    records["normalized_text"] = records["text"].map(normalize_text)
    records["text_length"] = records["text"].fillna("").astype(str).str.len()
    return records


def find_fuzzy_overlaps(
    records: pd.DataFrame,
    threshold: float = 0.88,
    high_threshold: float = 0.95,
    max_bigram_bucket: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Найти близкие тексты через редкие биграммы и ``SequenceMatcher``."""

    work = records.reset_index(drop=True).copy()
    work["_grams"] = work["normalized_text"].map(token_bigrams)
    inverted: dict[str, set[int]] = defaultdict(set)
    normalized_lookup: dict[str, list[int]] = defaultdict(list)
    for index, row in work.iterrows():
        normalized_lookup[row["normalized_text"]].append(index)
        for gram in row["_grams"]:
            inverted[gram].add(index)

    candidate_pairs: set[tuple[int, int]] = set()
    for indices in normalized_lookup.values():
        for position, left in enumerate(indices):
            candidate_pairs.update((left, right) for right in indices[position + 1 :])
    for index, grams in work["_grams"].items():
        counts: Counter[int] = Counter()
        for gram in grams:
            bucket = inverted[gram]
            if len(bucket) <= max_bigram_bucket:
                counts.update(bucket)
        for other, shared in counts.items():
            if other > index and shared >= 2:
                candidate_pairs.add((index, other))

    rows: list[dict[str, Any]] = []
    evaluated = 0
    for left_index, right_index in sorted(candidate_pairs):
        left, right = work.loc[left_index], work.loc[right_index]
        category = pair_category(left, right)
        if category is None:
            continue
        left_text, right_text = left["normalized_text"], right["normalized_text"]
        if not left_text or not right_text:
            continue
        length_ratio = min(len(left_text), len(right_text)) / max(len(left_text), len(right_text))
        if length_ratio < 0.75:
            continue
        evaluated += 1
        score = SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()
        if score < threshold:
            continue
        rows.append(
            {
                "pair_category": category,
                "left_id": left["source_id"],
                "right_id": right["source_id"],
                "left_source": left["source"],
                "right_source": right["source"],
                "left_split": left["split"],
                "right_split": right["split"],
                "similarity": round(score, 6),
                "similarity_ge_095": int(score >= high_threshold),
                "left_length": int(left["text_length"]),
                "right_length": int(right["text_length"]),
                "left_text": left["text"],
                "right_text": right["text"],
            }
        )
    columns = [
        "pair_category",
        "left_id",
        "right_id",
        "left_source",
        "right_source",
        "left_split",
        "right_split",
        "similarity",
        "similarity_ge_095",
        "left_length",
        "right_length",
        "left_text",
        "right_text",
    ]
    report = pd.DataFrame(rows, columns=columns)
    summary = {
        "records": len(work),
        "threshold": threshold,
        "high_threshold": high_threshold,
        "candidate_pairs_after_blocking": len(candidate_pairs),
        "candidate_pairs_evaluated": evaluated,
        "match_rows": len(report),
        "counts_by_category": {
            category: int((report["pair_category"] == category).sum())
            for category in PAIR_CATEGORIES
        },
        "counts_ge_095": {
            category: int(
                (
                    report["pair_category"].eq(category)
                    & report["similarity_ge_095"].eq(1)
                ).sum()
            )
            for category in PAIR_CATEGORIES
        },
        "method": (
            "NFKC/casefold/punctuation normalization; exact normalized groups; "
            "rare token-bigram blocking with at least two shared bigrams; "
            "length ratio >= 0.75; difflib.SequenceMatcher"
        ),
        "limitations": (
            "Эвристический blocking может пропустить семантические перефразы и короткие "
            "типовые отзывы без двух общих биграмм; совпадения требуют ручной проверки."
        ),
    }
    return report, summary


def read_existing(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def assert_no_manual_answers(output: Path) -> None:
    specs = {
        "weak_label_manual_check_50.csv": ("manual_sentiment", "manual_notes"),
        "brand_names_review.csv": ("manual_brand_check", "manual_notes"),
        "brand_records_review_30.csv": ("manual_brand_check", "manual_notes"),
    }
    for filename, columns in specs.items():
        path = output / filename
        if not path.is_file():
            continue
        existing = read_existing(path)
        missing = set(columns) - set(existing.columns)
        if missing:
            raise ValueError(f"В {path} отсутствуют поля: {sorted(missing)}")
        if any(existing[column].astype(str).str.strip().ne("").any() for column in columns):
            raise FileExistsError(
                f"В {path} найдены ручные ответы; подготовка остановлена без перезаписи"
            )


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def safe_write_template(path: Path, expected: pd.DataFrame, manual_columns: tuple[str, ...]) -> str:
    payload = dataframe_csv_bytes(expected)
    if not path.exists():
        path.write_bytes(payload)
        return "created"
    existing = read_existing(path)
    if list(existing.columns) != list(expected.columns):
        raise ValueError(f"Структура существующего шаблона отличается: {path}")
    if any(existing[column].str.strip().ne("").any() for column in manual_columns):
        raise FileExistsError(f"В {path} найдены ручные ответы")
    expected_text = expected.fillna("").astype(str)
    if not existing.equals(expected_text):
        raise ValueError(f"Существующий пустой шаблон не совпадает с воспроизводимым: {path}")
    return "unchanged"


def write_bytes_if_changed(path: Path, payload: bytes) -> str:
    if path.is_file() and path.read_bytes() == payload:
        return "unchanged"
    path.write_bytes(payload)
    return "written"


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    quality = config["quality_checks"]
    seed = int(quality["random_seed"])
    sentiment_quotas = {
        str(brand): int(count)
        for brand, count in quality["sentiment_train_quotas_per_label"].items()
    }
    brand_quotas = {
        str(brand): int(count) for brand, count in quality["brand_record_quotas"].items()
    }
    original = pd.read_csv(args.original.resolve())
    manual = pd.read_csv(args.manual.resolve())
    pool = pd.read_parquet(args.pool.resolve())
    sample = pd.read_csv(args.sample.resolve())

    annotation, reference = select_sentiment_reviews(original, sentiment_quotas, seed + 50)
    names_review = build_brand_names_review(pool)
    records_review = select_brand_records(sample, brand_quotas, seed + 30)
    fuzzy_report, fuzzy_summary = find_fuzzy_overlaps(
        fuzzy_records(original, manual),
        threshold=float(quality["fuzzy_threshold"]),
        high_threshold=float(quality["fuzzy_high_threshold"]),
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert_no_manual_answers(output)
    statuses = {
        "weak_label_manual_check_50.csv": safe_write_template(
            output / "weak_label_manual_check_50.csv",
            annotation,
            ("manual_sentiment", "manual_notes"),
        ),
        "weak_label_reference_50.csv": write_bytes_if_changed(
            output / "weak_label_reference_50.csv", dataframe_csv_bytes(reference)
        ),
        "brand_names_review.csv": safe_write_template(
            output / "brand_names_review.csv",
            names_review,
            ("manual_brand_check", "manual_notes"),
        ),
        "brand_records_review_30.csv": safe_write_template(
            output / "brand_records_review_30.csv",
            records_review,
            ("manual_brand_check", "manual_notes"),
        ),
        "fuzzy_overlap_report.csv": write_bytes_if_changed(
            output / "fuzzy_overlap_report.csv", dataframe_csv_bytes(fuzzy_report)
        ),
    }
    fuzzy_payload = json.dumps(fuzzy_summary, ensure_ascii=False, indent=2).encode("utf-8")
    statuses["fuzzy_overlap_summary.json"] = write_bytes_if_changed(
        output / "fuzzy_overlap_summary.json", fuzzy_payload
    )
    manifest = {
        "status": "prepared_awaiting_manual_review",
        "random_seed": seed,
        "sentiment_selection_seed": seed + 50,
        "brand_record_selection_seed": seed + 30,
        "sentiment_train_quotas_per_label": sentiment_quotas,
        "brand_record_quotas": brand_quotas,
        "sentiment_rows": len(annotation),
        "brand_name_rows": len(names_review),
        "brand_record_rows": len(records_review),
        "fuzzy": fuzzy_summary,
        "source_paths": {
            "original": str(args.original.resolve()),
            "manual": str(args.manual.resolve()),
            "available_pool": str(args.pool.resolve()),
            "sample": str(args.sample.resolve()),
        },
        "file_sha256": {
            filename: sha256(output / filename)
            for filename in (
                "weak_label_manual_check_50.csv",
                "weak_label_reference_50.csv",
                "brand_names_review.csv",
                "brand_records_review_30.csv",
                "fuzzy_overlap_report.csv",
                "fuzzy_overlap_summary.json",
            )
        },
        "manual_fields_empty": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    statuses["manifest.json"] = write_bytes_if_changed(output / "manifest.json", manifest_payload)
    print(json.dumps({"manifest": manifest, "write_status": statuses}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
