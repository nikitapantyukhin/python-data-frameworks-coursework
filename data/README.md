# Локальные данные

Исходные и производные датасеты исключены из Git. Репозиторий хранит только пустые каталоги, код, агрегированные отчёты и графики.

```text
data/
├── raw/
│   ├── geo_reviews/
│   ├── hagrid/
│   └── cn_gesture_recognition/
├── processed/
└── synthetic/
```

## Geo Reviews и Otzyvru

Исходный Parquet ожидается в `data/raw/geo_reviews/data/`. Результаты подготовки записываются в `data/processed/geo_fastfood_brands/` и `data/processed/fastfood_experiment/`. Полные тексты и ручные аннотации Otzyvru хранятся в `.local/fastfood_annotation/`.

```bash
python -m scripts.geo_reviews.download_geo_reviews
python -m scripts.geo_reviews.validate_geo_reviews
python -m scripts.fastfood.prepare_geo_brand_sample
python -m scripts.fastfood.prepare_fastfood_brand_splits
python -m scripts.fastfood.validate_fastfood_brands
python -m scripts.fastfood.eda_fastfood_brands
python -m scripts.fastfood.quality.audit_fastfood_annotations
python -m scripts.fastfood.quality.check_review_overlap
```

Материалы ручного контроля основной тысячи находятся в `.local/geo_brands_quality_checks/` и проверяются командами:

```bash
python -m scripts.fastfood.quality.prepare_geo_brands_quality_checks
python -m scripts.fastfood.quality.audit_geo_brands_quality_checks
```

## HaGRIDv2 и дополнительные изображения

Подмножество HaGRIDv2 ожидается в `data/raw/hagrid/`. Исходные дополнительные изображения [CNGR/CN_Gesture_Recognition](https://huggingface.co/datasets/CNGR/CN_Gesture_Recognition) при воспроизведении размещаются в `data/raw/cn_gesture_recognition/` с исходными папками классов. Подготовленные 200 изображений и служебные manifests находятся в `data/processed/hagrid_manual_enrichment/`. Два экспорта Label Studio сохранены как `data/01_participant_1_processed.json` и `data/02_participant_2_processed.json`.

Локальный `manifests/manual_selection_manifest.csv` фиксирует ручной отбор, контрольные суммы и размеры файлов. Скрипт подготовки использует этот manifest и создаёт только пустые задачи Label Studio; готовая детекционная разметка источника не требуется.

```bash
python -m scripts.hagrid.validate_hagrid
python -m scripts.hagrid.prepare_manual_enrichment
python -m scripts.hagrid.validate_manual_enrichment
python -m scripts.hagrid.analyze_label_studio_exports \
  --participant-1 data/01_participant_1_processed.json \
  --participant-2 data/02_participant_2_processed.json
```

Каталог `data/synthetic/` зарезервирован для будущего этапа. Искусственные данные пока не создавались.
