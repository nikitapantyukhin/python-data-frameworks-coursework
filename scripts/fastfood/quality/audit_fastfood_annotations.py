"""Проверить локальный комплект независимой разметки отзывов Otzyvru.

Скрипт не меняет исходные CSV. Он автоматически находит комплект даже при
одном дополнительном уровне вложенности и сохраняет только локальные отчёты.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from zipfile import BadZipFile, ZipFile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.fastfood.fastfood_data import normalize_text


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = ROOT / ".local" / "fastfood_annotation"
REQUIRED_FILES = {
    "participant_1": "01_participant_1.csv",
    "participant_2": "02_participant_2.csv",
    "comparison": "03_comparison.csv",
    "final": "04_final_dataset.csv",
    "provenance": "05_provenance.csv",
}
SUPPORT_FILES = (
    "annotation_report.xlsx",
    "annotation_summary.md",
    "codebook.md",
    "verify_annotations.py",
    "README.txt",
)
ALIASES = {
    "id": ("id", "review_id", "task_id", "record_id", "source_id"),
    "text": ("text", "review", "review_text", "content"),
    "label": ("label", "sentiment", "annotation", "class", "final_label"),
    "network": ("network", "chain", "brand", "restaurant", "organization"),
    "rating": ("rating", "source_rating", "original_rating"),
    "url": ("url", "source_url", "review_url", "page_url", "source_page_url"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--expected-count", type=int, default=200)
    return parser.parse_args()


def locate_bundle(root: Path) -> Path:
    root = root.expanduser().resolve()
    matches = sorted(root.rglob(REQUIRED_FILES["final"])) if root.exists() else []
    complete = [
        path.parent
        for path in matches
        if all((path.parent / name).is_file() for name in REQUIRED_FILES.values())
    ]
    if len(complete) != 1:
        raise FileNotFoundError(
            f"Ожидался один полный комплект в {root}, найдено: {len(complete)}. "
            f"Требуются: {', '.join(REQUIRED_FILES.values())}"
        )
    return complete[0]


def find_column(frame: pd.DataFrame, field: str, *, required: bool = True) -> str | None:
    lower = {str(column).strip().casefold(): str(column) for column in frame.columns}
    for alias in ALIASES[field]:
        if alias.casefold() in lower:
            return lower[alias.casefold()]
    if required:
        raise ValueError(
            f"Не найдено поле {field!r}; столбцы: {', '.join(map(str, frame.columns))}"
        )
    return None


def canonical(frame: pd.DataFrame, *, need_label: bool = True) -> pd.DataFrame:
    id_col = find_column(frame, "id")
    text_col = find_column(frame, "text")
    result = pd.DataFrame(
        {
            "id": frame[id_col].astype(str).str.strip(),
            "text": frame[text_col].fillna("").astype(str),
        }
    )
    if need_label:
        label_col = find_column(frame, "label")
        result["label"] = pd.to_numeric(frame[label_col], errors="coerce")
    return result


def cohens_kappa(left: pd.Series, right: pd.Series) -> float:
    left = left.astype(int)
    right = right.astype(int)
    observed = float(left.eq(right).mean())
    left_p = left.value_counts(normalize=True)
    right_p = right.value_counts(normalize=True)
    expected = sum(float(left_p.get(label, 0) * right_p.get(label, 0)) for label in (0, 1))
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def tracked_text_leaks(texts: pd.Series) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = [item for item in completed.stdout.decode().split("\0") if item]
    # Короткие общие фразы не дают надёжного сигнала; их контролирует provenance.
    needles = {text for text in texts.dropna().astype(str) if len(text.strip()) >= 20}
    leaks: list[str] = []
    for relative in paths:
        path = ROOT / relative
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if any(needle in content for needle in needles):
            leaks.append(relative)
    return leaks


def provenance_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": len(frame), "columns": list(map(str, frame.columns))}
    for field in ("network", "rating", "url"):
        column = find_column(frame, field, required=False)
        if column is None:
            result[field] = {"column_present": False, "missing": len(frame)}
            continue
        values = frame[column]
        missing = int(
            (values.isna() | values.fillna("").astype(str).str.strip().eq("")).sum()
        )
        field_result: dict[str, Any] = {
            "column_present": True,
            "missing": missing,
            "unique": int(values.nunique(dropna=True)),
        }
        if field in {"network", "rating", "url"}:
            field_result["counts"] = {
                str(key): int(value) for key, value in values.value_counts(dropna=False).items()
            }
        result[field] = field_result
    for column in ("source_site", "source_row", "collection_date"):
        if column not in frame:
            result[column] = {"column_present": False, "missing": len(frame)}
            continue
        values = frame[column]
        result[column] = {
            "column_present": True,
            "missing": int(
                (values.isna() | values.fillna("").astype(str).str.strip().eq("")).sum()
            ),
            "unique": int(values.nunique(dropna=True)),
        }
    network_col = find_column(frame, "network", required=False)
    if network_col is not None and "source_row" in frame:
        pair_duplicates = int(frame.duplicated([network_col, "source_row"]).sum())
        result["network_source_row_pairs"] = {
            "unique": int(frame[[network_col, "source_row"]].drop_duplicates().shape[0]),
            "duplicate_extra": pair_duplicates,
        }
    return result


def metadata_summary(bundle: Path, requested_root: Path) -> dict[str, str]:
    candidates = (bundle / "annotation_metadata.md", requested_root / "annotation_metadata.md")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return {"status": "Не указано"}
    result: dict[str, str] = {"file": str(path)}
    section = "Документ"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        cleaned = value.strip().strip(".").strip("`")
        result[f"{section} / {key.strip()}"] = (
            "Не указано" if cleaned == "ЗАПОЛНИТЬ" else cleaned
        )
    return result


def xlsx_summary(path: Path) -> dict[str, Any]:
    """Структурно проверить XLSX без извлечения текстов и без openpyxl."""

    required = {"[Content_Types].xml", "xl/workbook.xml"}
    try:
        with ZipFile(path) as archive:
            names = set(archive.namelist())
            missing = sorted(required - names)
            corrupt_member = archive.testzip()
            return {
                "readable": not missing and corrupt_member is None,
                "missing_required_members": missing,
                "corrupt_member": corrupt_member,
                "worksheet_files": len(
                    [name for name in names if name.startswith("xl/worksheets/sheet")]
                ),
            }
    except (BadZipFile, OSError) as error:
        return {"readable": False, "error": f"{type(error).__name__}: {error}"}


def main() -> int:
    args = parse_args()
    try:
        bundle = locate_bundle(args.annotation_root)
    except FileNotFoundError as error:
        print(f"[ERROR] {error}")
        return 2
    missing_support = [name for name in SUPPORT_FILES if not (bundle / name).is_file()]
    if missing_support:
        print(f"[ERROR] Не найдены вспомогательные файлы: {', '.join(missing_support)}")
        return 2
    provided_verifier = subprocess.run(
        [sys.executable, "verify_annotations.py"],
        cwd=bundle,
        text=True,
        capture_output=True,
    )
    raw = {
        key: pd.read_csv(bundle / filename)
        for key, filename in REQUIRED_FILES.items()
    }
    p1 = canonical(raw["participant_1"])
    p2 = canonical(raw["participant_2"])
    final = canonical(raw["final"])
    errors: list[str] = []
    if provided_verifier.returncode:
        errors.append(
            "Комплектный verify_annotations.py завершился с кодом "
            f"{provided_verifier.returncode}"
        )
    comparison = raw["comparison"]
    if len(comparison) != args.expected_count:
        errors.append(
            f"comparison: ожидалось {args.expected_count}, найдено {len(comparison)}"
        )

    for name, frame in (("participant_1", p1), ("participant_2", p2), ("final", final)):
        if len(frame) != args.expected_count:
            errors.append(f"{name}: ожидалось {args.expected_count}, найдено {len(frame)}")
        if frame["id"].duplicated().any():
            errors.append(f"{name}: повторяющиеся ID")
        if frame["text"].str.strip().eq("").any():
            errors.append(f"{name}: пустые тексты")
        if frame["label"].isna().any() or not set(frame["label"].dropna()).issubset({0, 1}):
            errors.append(f"{name}: метки должны быть только 0/1 без пропусков")

    if set(p1["id"]) != set(p2["id"]) or set(p1["id"]) != set(final["id"]):
        errors.append("Наборы ID участников и финального датасета различаются")
    joined = p1.merge(p2, on="id", suffixes=("_p1", "_p2"), validate="one_to_one")
    joined = joined.merge(
        final[["id", "text", "label"]].rename(
            columns={"text": "text_final", "label": "label_final"}
        ),
        on="id",
        validate="one_to_one",
    )
    text_mismatch = ~(
        joined["text_p1"].eq(joined["text_p2"])
        & joined["text_p1"].eq(joined["text_final"])
    )
    if text_mismatch.any():
        errors.append(f"Тексты различаются по ID в {int(text_mismatch.sum())} строках")

    normalized = final["text"].map(normalize_text)
    exact_duplicate_extra = int(final["text"].duplicated().sum())
    lengths = final["text"].str.len()
    duplicate_extra = int(normalized.duplicated().sum())
    if duplicate_extra:
        errors.append(f"Нормализованные дубликаты: {duplicate_extra} лишних строк")
    disagreements = joined.loc[joined["label_p1"].ne(joined["label_p2"]), "id"].tolist()
    if disagreements != ["15"]:
        errors.append(f"Ожидалось единственное расхождение ID 15, найдено: {disagreements}")
    id15 = joined.loc[joined["id"].eq("15"), "label_final"]
    if len(id15) != 1 or int(id15.iloc[0]) != 1:
        errors.append("Итоговая метка ID 15 должна быть равна 1")

    agreement = float(joined["label_p1"].eq(joined["label_p2"]).mean())
    kappa = cohens_kappa(joined["label_p1"], joined["label_p2"])
    final_balance = {str(int(k)): int(v) for k, v in final["label"].value_counts().items()}
    if final_balance != {"0": 100, "1": 100}:
        errors.append(f"Ожидался итоговый баланс 100/100, найдено: {final_balance}")
    if abs(agreement - 0.995) > 1e-12:
        errors.append(f"Percentage Agreement должен быть 0.995, найдено {agreement}")
    if not 0.989 <= kappa <= 0.991:
        errors.append(f"Cohen's Kappa ожидалась около 0.990, найдено {kappa}")

    report = {
        "bundle": str(bundle),
        "status": "FAILED" if errors else "PASSED",
        "errors": errors,
        "provided_verifier": {
            "returncode": provided_verifier.returncode,
            "stdout": provided_verifier.stdout,
            "stderr": provided_verifier.stderr,
        },
        "rows": len(final),
        "unique_ids": int(final["id"].nunique()),
        "blank_texts": int(final["text"].str.strip().eq("").sum()),
        "exact_duplicate_extra": exact_duplicate_extra,
        "normalized_duplicate_extra": duplicate_extra,
        "length_chars": {
            "min": int(lengths.min()),
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "p95": float(lengths.quantile(0.95)),
            "max": int(lengths.max()),
            "unusually_short_ids_le_10_chars": final.loc[lengths.le(10), "id"].tolist(),
            "unusually_long_ids_above_p99": final.loc[
                lengths.gt(lengths.quantile(0.99)), "id"
            ].tolist(),
        },
        "agreement_count": int(joined["label_p1"].eq(joined["label_p2"]).sum()),
        "disagreement_ids": disagreements,
        "percentage_agreement": agreement,
        "cohens_kappa": kappa,
        "final_balance": final_balance,
        "provenance": provenance_summary(raw["provenance"]),
        "annotation_metadata": metadata_summary(bundle, args.annotation_root.resolve()),
        "annotation_report_xlsx": xlsx_summary(bundle / "annotation_report.xlsx"),
        "tracked_text_leaks": tracked_text_leaks(final["text"]),
    }
    if not report["annotation_report_xlsx"]["readable"]:
        report["errors"].append("annotation_report.xlsx не прошёл структурную проверку")
        report["status"] = "FAILED"
    if report["tracked_text_leaks"]:
        report["errors"].append(
            f"Полные тексты найдены в Git-файлах: {report['tracked_text_leaks']}"
        )
        report["status"] = "FAILED"
    (bundle / "verification_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
