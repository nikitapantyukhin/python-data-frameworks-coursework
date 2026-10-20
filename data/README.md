# Локальные данные

Каталог `data/` предназначен для исходных и производных наборов данных. Содержимое подкаталогов `raw/`, `processed/` и `synthetic/` исключено из Git; в репозитории сохраняются только этот документ и `.gitkeep` для структуры каталогов.

## Geo Reviews Dataset 2023

Источник: [GitHub](https://github.com/yandex/geo-reviews-dataset-2023)
Parquet-публикация: [Hugging Face](https://huggingface.co/datasets/d0rj/geo-reviews-dataset-2023)

Из корня проекта:

```bash
source .venv/bin/activate
python scripts/download_geo_reviews.py
```

Ожидаемый файл:

```text
data/raw/geo_reviews/data/train-00000-of-00001-49261e4e5a35a5a0.parquet
```

Контрольные признаки версии: 170 880 716 байт, 500 000 строк, столбцы `address`, `name_ru`, `rating`, `rubrics`, `text`.

## Hard Hat Workers v10

Источник: [Roboflow Universe](https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/10)

Необходимо вручную скачать версию **10 — `raw_AllClasses`** в формате YOLOv8 и распаковать содержимое в `data/raw/hard_hat/`:

```text
data/raw/hard_hat/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

| Часть | Изображения | TXT-аннотации |
| --- | ---: | ---: |
| train | 4 916 | 4 916 |
| valid | 1 413 | 1 413 |
| test | 706 | 706 |
| **Итого** | **7 035** | **7 035** |
