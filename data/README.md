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
python scripts/download_geo_reviews.py
python scripts/validate_geo_reviews.py
```

Ожидаемый файл:

```text
data/raw/geo_reviews/data/train-00000-of-00001-49261e4e5a35a5a0.parquet
```

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
python scripts/validate_hagrid.py
```

Подробное описание состава данных и известных ограничений приведено в [описании датасетов](../docs/datasets.md).
