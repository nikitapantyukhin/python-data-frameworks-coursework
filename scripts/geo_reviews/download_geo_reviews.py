"""Загрузка проверенной публикации Geo Reviews Dataset 2023 в формате Parquet.

Скрипт запускается из корня проекта командой
``python -m scripts.geo_reviews.download_geo_reviews``. Если файл ожидаемого размера уже
существует, он используется повторно без сетевого запроса.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "raw" / "geo_reviews"
REPO_ID = "d0rj/geo-reviews-dataset-2023"
FILENAME = "data/train-00000-of-00001-49261e4e5a35a5a0.parquet"
EXPECTED_SIZE = 170_880_716


def main() -> None:
    """Загрузить файл для EDA и проверить его размер в байтах."""

    DEST.mkdir(parents=True, exist_ok=True)
    expected_path = DEST / FILENAME
    if expected_path.exists() and expected_path.stat().st_size == EXPECTED_SIZE:
        print(f"Already available: {expected_path}")
        return

    print("Downloading Geo Reviews dataset (~171 MB) from Hugging Face...")
    downloaded_path = Path(hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=FILENAME,
        local_dir=DEST,
    ))
    actual_size = downloaded_path.stat().st_size
    if actual_size != EXPECTED_SIZE:
        raise RuntimeError(
            f"Unexpected file size: {actual_size} bytes; expected {EXPECTED_SIZE}. "
            "Check the dataset revision before running the EDA."
        )
    print(f"Ready: {downloaded_path}")


if __name__ == "__main__":
    main()
