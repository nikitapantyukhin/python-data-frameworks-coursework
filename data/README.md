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
```

Ожидаемый файл:

```text
data/raw/geo_reviews/data/train-00000-of-00001-49261e4e5a35a5a0.parquet
```

Основная выборка Geo Reviews находится на этапе пересмотра методики
формирования. В `data/processed/` сейчас нет действующей основной текстовой
выборки или разбиений; команда их подготовки будет добавлена после утверждения
новых правил.

Полные тексты дополнительных отзывов и результаты независимой разметки
размещаются в `.local/fastfood_annotation/`. Исторические файлы ручного
контроля прежней Geo-выборки сохранены в `.local/geo_quality_checks/`; они не
являются входом для ещё не созданной новой выборки.

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
