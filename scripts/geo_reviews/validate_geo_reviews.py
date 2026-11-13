"""Валидация файла Geo Reviews в формате Parquet без изменения данных.

Скрипт запускается из корня проекта командой
``python -m scripts.geo_reviews.validate_geo_reviews``. Предупреждения описывают допустимые
несовершенства данных и не приводят к ненулевому коду завершения.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "data" / "raw" / "geo_reviews"
PARQUET_RELATIVE_PATH = Path("data/train-00000-of-00001-49261e4e5a35a5a0.parquet")
REQUIRED_COLUMNS = ("address", "name_ru", "rating", "rubrics", "text")
TEXT_COLUMNS = ("address", "name_ru", "rubrics", "text")
EXPECTED_ROWS = 500_000
EXPECTED_RATINGS = {0, 1, 2, 3, 4, 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Каталог Geo Reviews или файл Parquet "
            f"(каталог по умолчанию: {DEFAULT_DATA_ROOT})"
        ),
    )
    return parser.parse_args()


def parquet_path(data_root: Path) -> Path:
    """Определить путь к указанному файлу Parquet или стандартному расположению."""

    data_root = data_root.expanduser().resolve()
    return (
        data_root
        if data_root.suffix.lower() in {".parquet", ".pq"}
        else data_root / PARQUET_RELATIVE_PATH
    )


def is_text_column(series: pd.Series) -> bool:
    """Проверить, что каждое непустое значение является строкой Python."""

    present = series.dropna()
    return bool(present.map(lambda value: isinstance(value, str)).all())


def main() -> int:
    args = parse_args()
    source = parquet_path(args.data_root)
    errors: list[str] = []
    warnings: list[str] = []

    print("Geo Reviews validation")
    print(f"Data file: {source}")
    if not source.is_file():
        print(f"[ERROR] Parquet file does not exist: {source}")
        print("RESULT: FAILED (1 critical error, 0 warnings)")
        return 1

    try:
        frame = pd.read_parquet(source)
    except Exception as error:  # pandas и pyarrow используют разные исключения чтения
        print(f"[ERROR] Cannot read Parquet: {type(error).__name__}: {error}")
        print("RESULT: FAILED (1 critical error, 0 warnings)")
        return 1

    print(f"[INFO] Readable Parquet: {len(frame):,} rows, {len(frame.columns)} columns")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    extra_columns = [column for column in frame.columns if column not in REQUIRED_COLUMNS]
    if extra_columns:
        warnings.append(f"Extra columns: {', '.join(map(str, extra_columns))}")

    if errors:
        for message in errors:
            print(f"[ERROR] {message}")
        for message in warnings:
            print(f"[WARNING] {message}")
        print(f"RESULT: FAILED ({len(errors)} critical errors, {len(warnings)} warnings)")
        return 1

    if len(frame) == 0:
        errors.append("Dataset has no rows")
    elif len(frame) != EXPECTED_ROWS:
        warnings.append(f"Expected {EXPECTED_ROWS:,} rows, found {len(frame):,}")

    invalid_text_types = [
        column for column in TEXT_COLUMNS if not is_text_column(frame[column])
    ]
    if invalid_text_types:
        errors.append(f"Non-string values in text columns: {', '.join(invalid_text_types)}")
    if not pd.api.types.is_numeric_dtype(frame["rating"]):
        errors.append(f"rating must be numeric, found dtype {frame['rating'].dtype}")

    missing_counts = frame[list(REQUIRED_COLUMNS)].isna().sum().astype(int)
    print("[INFO] Missing values:")
    for column, count in missing_counts.items():
        print(f"  {column}: {count:,}")
        if count:
            warnings.append(f"{column} contains {count:,} missing values")

    present_text = frame["text"].notna()
    blank_texts = int(
        (present_text & frame["text"].fillna("").astype(str).str.strip().eq("")).sum()
    )
    print(f"[INFO] Blank non-null texts: {blank_texts:,}")
    if blank_texts:
        warnings.append(f"text contains {blank_texts:,} blank non-null values")
    if int(present_text.sum()) - blank_texts == 0:
        errors.append("No usable non-blank texts")

    if pd.api.types.is_numeric_dtype(frame["rating"]):
        non_integer_ratings = frame["rating"].dropna().mod(1).ne(0)
        if bool(non_integer_ratings.any()):
            errors.append(f"rating contains {int(non_integer_ratings.sum()):,} non-integer values")
        actual_ratings = {int(value) for value in frame["rating"].dropna().unique()}
        unexpected_ratings = sorted(actual_ratings - EXPECTED_RATINGS)
        print(f"[INFO] Rating values: {sorted(actual_ratings)}")
        if unexpected_ratings:
            warnings.append(f"Unexpected rating values: {unexpected_ratings}")

    full_duplicate_rows = int(frame.duplicated(keep=False).sum())
    full_duplicate_extra = int(frame.duplicated().sum())
    repeated_text_rows = int(frame.duplicated("text", keep=False).sum())
    repeated_text_extra = int(frame.duplicated("text").sum())
    repeated_text_values = int(
        frame.loc[frame.duplicated("text", keep=False), "text"].nunique(dropna=True)
    )
    print(
        "[INFO] Duplicates: "
        f"full rows={full_duplicate_rows:,} (extra={full_duplicate_extra:,}); "
        f"repeated-text rows={repeated_text_rows:,} "
        f"(extra={repeated_text_extra:,}, unique repeated texts={repeated_text_values:,})"
    )
    if full_duplicate_extra:
        warnings.append(f"Full duplicate extra rows: {full_duplicate_extra:,}")
    if repeated_text_extra:
        warnings.append(f"Repeated-text extra rows: {repeated_text_extra:,}")

    negative_count = int(frame["rating"].isin([1, 2]).sum())
    positive_count = int(frame["rating"].isin([4, 5]).sum())
    print(
        "[INFO] Preliminary binary labels: "
        f"negative={negative_count:,}, positive={positive_count:,}"
    )
    if not negative_count or not positive_count:
        errors.append("One of the preliminary binary classes is empty")
    else:
        imbalance = max(negative_count, positive_count) / min(negative_count, positive_count)
        print(f"[INFO] Preliminary class imbalance ratio: {imbalance:.2f}:1")
        if imbalance >= 3:
            warnings.append(f"Strong preliminary class imbalance: {imbalance:.2f}:1")

    for message in errors:
        print(f"[ERROR] {message}")
    for message in warnings:
        print(f"[WARNING] {message}")
    status = "FAILED" if errors else "PASSED"
    print(f"RESULT: {status} ({len(errors)} critical errors, {len(warnings)} warnings)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
