# Организация данных

Каталог `data/` предназначен для исходных, подготовленных и искусственно созданных данных. Содержимое датасетов исключено из Git.

```text
data/
├── raw/
│   ├── geo_reviews/
│   └── hagrid/
├── processed/
└── synthetic/
```

## Geo Reviews Dataset 2023

Исходные данные загружаются из Parquet-публикации датасета:

```bash
python -m scripts.geo_reviews.download_geo_reviews
python -m scripts.geo_reviews.validate_geo_reviews
python -m scripts.fastfood.prepare_geo_fastfood
```

Ожидаемый файл:

```text
data/raw/geo_reviews/data/train-00000-of-00001-49261e4e5a35a5a0.parquet
```

Локальные рабочие варианты создаются в `data/processed/geo_fastfood/`. Полные
тексты дополнительных отзывов и результаты разметки размещаются в
`.local/fastfood_annotation/`; оба каталога исключены из Git.

Файлы для ручного контроля слабых меток и предметного фильтра создаются командой
`python -m scripts.fastfood.quality.prepare_geo_quality_checks` в
`.local/geo_quality_checks/`.
Заполненные результаты проверяются без изменения данных командой
`python -m scripts.fastfood.quality.audit_geo_quality_checks`.

## HaGRIDv2

Используемое подмножество основано на [HaGRIDv2](https://github.com/hukenovs/hagrid) и публикации [GestureDetectionConnoisseurs/hagrid_subsets](https://huggingface.co/datasets/GestureDetectionConnoisseurs/hagrid_subsets). Исходные данные должны быть размещены в `data/raw/hagrid/`.

Ожидаемая структура:

```text
data/raw/hagrid/
├── call/
├── ...
├── xsign/
├── annotations.json
├── minimal-annotations.json
└── exported_images.json
```

Проверка структуры, изображений и аннотаций:

```bash
python -m scripts.hagrid.validate_hagrid
```

Подробное описание состава данных и известных ограничений приведено в [описании датасетов](../docs/datasets.md).
