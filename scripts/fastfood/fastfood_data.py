"""Общие операции подготовки текстов первого датасета."""

from __future__ import annotations

import hashlib
import re
import unicodedata

import pandas as pd


SPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


def normalize_text(value: object) -> str:
    """Нормализовать текст для дедупликации, не меняя исходное поле."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return SPACE_RE.sub(" ", NON_WORD_RE.sub(" ", text)).strip()


def stable_hash(value: object, seed: int) -> str:
    """Вернуть устойчивый ключ для воспроизводимой сортировки."""

    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def rubric_set(value: object) -> set[str]:
    """Разобрать строку рубрик на точные непустые компоненты."""

    return {part.strip() for part in str(value).split(";") if part.strip()}


def weak_label_from_rating(value: object) -> int:
    """Преобразовать рейтинг в слабую бинарную метку или ``-1``."""

    if value is None or pd.isna(value):
        return -1
    if value in (1, 2):
        return 0
    if value in (4, 5):
        return 1
    return -1


def token_bigrams(normalized: str) -> set[str]:
    """Получить биграммы слов для поиска близких текстов."""

    tokens = normalized.split()
    return {f"{left}\0{right}" for left, right in zip(tokens, tokens[1:])}
