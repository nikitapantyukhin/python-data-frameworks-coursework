"""Общие безопасные операции для подготовки текстов первого датасета."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable

import pandas as pd


SPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


def normalize_text(value: object) -> str:
    """Нормализовать текст только для дедупликации, не меняя исходное поле."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return SPACE_RE.sub(" ", NON_WORD_RE.sub(" ", text)).strip()


def stable_hash(value: object, seed: int) -> str:
    """Вернуть устойчивый ключ для воспроизводимой сортировки."""

    payload = f"{seed}\0{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rubric_set(value: object) -> set[str]:
    """Разобрать строку рубрик на точные непустые компоненты."""

    return {part.strip() for part in str(value).split(";") if part.strip()}


def serialize_rubric_set(values: set[str]) -> str:
    """Сериализовать множество рубрик в стабильном порядке."""

    return ";".join(sorted(values))


def csv_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Подготовить копию таблицы с рубриками к CSV-экспорту."""

    exported = frame.copy()
    if "rubric_set" in exported:
        exported["rubric_set"] = exported["rubric_set"].map(serialize_rubric_set)
    return exported


def weak_label_from_rating(value: object) -> int:
    """Преобразовать рейтинг в слабую бинарную метку или ``-1``."""

    if value is None or pd.isna(value):
        return -1
    if value in (1, 2):
        return 0
    if value in (4, 5):
        return 1
    return -1


def balanced_sample(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Детерминированно отобрать равное число строк классов 0 и 1."""

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


def assign_splits(
    frame: pd.DataFrame,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> pd.DataFrame:
    """Назначить воспроизводимые стратифицированные части по ``group_id``."""

    if validation_fraction + test_fraction >= 1:
        raise ValueError("Сумма долей validation и test должна быть меньше 1")
    result = frame.copy()
    result["split"] = "train"
    for _, subset in result.groupby("label", sort=True):
        ordered = subset.assign(
            _key=subset["group_id"].map(lambda value: stable_hash(value, seed))
        ).sort_values("_key")
        count = len(ordered)
        test_count = round(count * test_fraction)
        validation_count = round(count * validation_fraction)
        result.loc[ordered.index[:test_count], "split"] = "test"
        result.loc[
            ordered.index[test_count : test_count + validation_count], "split"
        ] = "validation"
    return result


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


def token_bigrams(normalized: str) -> set[str]:
    """Получить биграммы слов для экономного поиска близких текстов."""

    tokens = normalized.split()
    return {f"{left}\0{right}" for left, right in zip(tokens, tokens[1:])}


def ensure_disjoint(groups: Iterable[tuple[str, Iterable[str]]]) -> None:
    """Проверить, что нормализованные тексты не пересекаются между частями."""

    seen: dict[str, str] = {}
    for group_name, texts in groups:
        for text in texts:
            key = normalize_text(text)
            if not key:
                raise ValueError(f"Пустой нормализованный текст в {group_name}")
            previous = seen.get(key)
            if previous is not None and previous != group_name:
                raise ValueError(
                    f"Утечка нормализованного текста между {previous} и {group_name}"
                )
            seen[key] = group_name
