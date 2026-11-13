"""Общие безопасные операции для подготовки текстов первого датасета."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable


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
