"""Найти точные, нормализованные и близкие пересечения Otzyvru/Geo Reviews."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from scripts.fastfood.fastfood_data import normalize_text, token_bigrams
from scripts.fastfood.quality.audit_fastfood_annotations import canonical, locate_bundle
from scripts.geo_reviews.validate_geo_reviews import DEFAULT_DATA_ROOT, parquet_path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANNOTATIONS = ROOT / ".local" / "fastfood_annotation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.88)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = locate_bundle(args.annotation_root)
    manual = canonical(pd.read_csv(bundle / "04_final_dataset.csv"))
    manual["normalized"] = manual["text"].map(normalize_text)
    exact_lookup: dict[str, list[int]] = defaultdict(list)
    normalized_lookup: dict[str, list[int]] = defaultdict(list)
    bigram_lookup: dict[str, set[int]] = defaultdict(set)
    manual_bigrams: list[set[str]] = []
    for idx, row in manual.iterrows():
        exact_lookup[row["text"]].append(idx)
        normalized_lookup[row["normalized"]].append(idx)
        grams = token_bigrams(row["normalized"])
        manual_bigrams.append(grams)
        for gram in grams:
            bigram_lookup[gram].add(idx)

    source = parquet_path(args.data_root)
    geo = pd.read_parquet(source, columns=["text"])
    matches: list[dict] = []
    fuzzy_seen: set[tuple[int, int]] = set()
    for geo_idx, text in geo["text"].items():
        exact_ids = exact_lookup.get(text, [])
        normalized = normalize_text(text)
        normalized_ids = normalized_lookup.get(normalized, [])
        for manual_idx in set(exact_ids) | set(normalized_ids):
            matches.append(
                {
                    "manual_id": manual.at[manual_idx, "id"],
                    "geo_row": int(geo_idx),
                    "match_type": "exact" if manual_idx in exact_ids else "normalized",
                    "score": 1.0,
                }
            )
        if len(normalized) < 20 or normalized_ids:
            continue
        grams = token_bigrams(normalized)
        candidate_counts: Counter[int] = Counter()
        for gram in grams:
            # Очень общие биграммы не являются хорошими кандидатными ключами.
            candidate_ids = bigram_lookup.get(gram, set())
            if len(candidate_ids) <= 10:
                candidate_counts.update(candidate_ids)
        for manual_idx, shared in candidate_counts.items():
            pair = (manual_idx, int(geo_idx))
            if pair in fuzzy_seen or shared < 2:
                continue
            target = manual.at[manual_idx, "normalized"]
            length_ratio = min(len(target), len(normalized)) / max(len(target), len(normalized))
            if length_ratio < 0.75:
                continue
            union = grams | manual_bigrams[manual_idx]
            jaccard = len(grams & manual_bigrams[manual_idx]) / len(union) if union else 0
            if jaccard < 0.60:
                continue
            score = SequenceMatcher(None, target, normalized, autojunk=False).ratio()
            if score >= args.fuzzy_threshold:
                fuzzy_seen.add(pair)
                matches.append(
                    {
                        "manual_id": manual.at[manual_idx, "id"],
                        "geo_row": int(geo_idx),
                        "match_type": "fuzzy_candidate",
                        "score": round(score, 6),
                    }
                )

    result = pd.DataFrame(matches, columns=["manual_id", "geo_row", "match_type", "score"])
    result.to_csv(bundle / "geo_overlap_report.csv", index=False)
    summary = {
        "manual_rows": len(manual),
        "geo_rows": len(geo),
        "match_rows": len(result),
        "unique_manual_ids": int(result["manual_id"].nunique()) if len(result) else 0,
        "counts": result["match_type"].value_counts().to_dict() if len(result) else {},
        "fuzzy_threshold": args.fuzzy_threshold,
        "note": "fuzzy_candidate требует ручной проверки; строки автоматически не удаляются",
    }
    (bundle / "geo_overlap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
