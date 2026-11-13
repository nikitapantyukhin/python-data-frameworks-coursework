# Сравнительный анализ методов обогащения данных

Курсовой проект по дисциплине «Фреймворки для анализа данных на Питоне».

**Тема:** «Сравнительный анализ влияния ручного и искусственного обогащения данных на качество и устойчивость обучения моделей машинного обучения».

## Цель исследования

Цель проекта — сравнить ручное, искусственное и совместное обогащение данных при фиксированных условиях обучения и оценки. Исследование охватывает два типа данных: русскоязычные текстовые отзывы и изображения жестов рук.

## Участники

- Смирнов Константин Алексеевич;
- Пантюхин Никита Олегович;
- группа М8О-201БВ-25.

## Датасеты

| Датасет | Использование в проекте | Текущий объём |
| --- | --- | ---: |
| [Geo Reviews Dataset 2023](https://github.com/yandex/geo-reviews-dataset-2023) | Источник русскоязычных отзывов; подготовлены сбалансированные предварительные выборки фастфуда на 200, 500 и 1 000 строк | 500 000 исходных отзывов |
| [HaGRIDv2](https://github.com/hukenovs/hagrid) | Классификация и детекция жестов рук | Подмножество: 34 000 изображений, 34 класса |

Для HaGRIDv2 используются аннотации ограничивающих прямоугольников (bounding box, bbox), метки жестов и идентификаторы участников. Подробные сведения об источниках, составе и ограничениях приведены в [описании датасетов](docs/datasets.md).

## Состояние проекта

Реализованы загрузка и валидация Geo Reviews, предметный отбор фастфуда,
предварительные сбалансированные выборки и воспроизводимое разбиение варианта на
1 000 строк. Дополнительно собраны и независимо размечены двумя участниками 200
отзывов Otzyvru. Комплектная и независимая программные проверки подтвердили 199
совпадений, одно согласованное расхождение (ID 15), Percentage Agreement 99,5%,
Cohen's Kappa 0,990 и итоговый баланс 100/100. Полные тексты и построчные
материалы остаются в `.local/fastfood_annotation/`. Обучение моделей и
синтетическое обогащение не выполнялись.

## Структура репозитория

```text
.
├── configs/
│   └── fastfood_experiment.json # параметры будущего эксперимента
├── data/
│   ├── README.md
│   ├── raw/                  # исходные данные, исключены из Git
│   ├── processed/            # подготовленные данные, исключены из Git
│   └── synthetic/            # будущие искусственные данные, исключены из Git
├── docs/
│   ├── datasets.md           # источники, лицензии и ограничения
│   ├── fastfood_annotation_methodology.md
│   └── methodology.md        # методология экспериментов
├── notebooks/
│   ├── 01_geo_reviews_eda.ipynb
│   └── 02_hagrid_eda.ipynb
├── reports/
│   ├── eda_summary.md
│   ├── fastfood_eda_summary.md
│   └── figures/
├── scripts/
│   ├── geo_reviews/          # загрузка и валидация Geo Reviews
│   ├── fastfood/             # подготовка, EDA и разбиения первого датасета
│   │   └── quality/          # контроль ручной разметки и качества
│   └── hagrid/               # валидация HaGRIDv2
├── tests/
│   └── test_fastfood_pipeline.py
├── README.md
└── requirements.txt
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Основные команды

Загрузка и валидация Geo Reviews:

```bash
python -m scripts.geo_reviews.download_geo_reviews
python -m scripts.geo_reviews.validate_geo_reviews
python -m scripts.fastfood.prepare_geo_fastfood
python -m scripts.fastfood.quality.prepare_geo_quality_checks
python -m scripts.fastfood.quality.audit_geo_quality_checks
python -m scripts.fastfood.quality.audit_fastfood_annotations
python -m scripts.fastfood.quality.check_review_overlap
python -m scripts.fastfood.eda_fastfood
python -m scripts.fastfood.prepare_fastfood_splits
```

Для изолированного запуска EDA пути графиков и JSON-сводки можно задать явно:

```bash
python -m scripts.fastfood.eda_fastfood \
  --figures-dir /tmp/fastfood_figures \
  --summary-output /tmp/fastfood_eda_summary.json
```

Валидация HaGRIDv2:

```bash
python -m scripts.hagrid.validate_hagrid
```

Для полного декодирования изображений вместо структурной проверки Pillow:

```bash
python -m scripts.hagrid.validate_hagrid --decode-images
```

Проверка функций подготовки первого датасета:

```bash
python -m unittest discover -s tests -v
```

Запуск ноутбуков:

```bash
jupyter lab
```

Исходные данные должны быть размещены согласно [инструкции по организации данных](data/README.md). Они не включаются в Git.

## Документация

- [Источники и характеристики датасетов](docs/datasets.md)
- [Методология исследования](docs/methodology.md)
- [Результаты разведочного анализа данных](reports/eda_summary.md)
- [Методика ручной разметки](docs/fastfood_annotation_methodology.md)
- [EDA и рабочие выборки фастфуда](reports/fastfood_eda_summary.md)
