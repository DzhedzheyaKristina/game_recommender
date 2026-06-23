# Рекомендательная система игр на основе Steam Reviews и LLM-реранжирования

## Описание проекта

Этот репозиторий содержит исследовательский прототип ВКР по рекомендательным системам для геймеров на основе текстов отзывов Steam.

Проект реализует:

- потоковую предобработку большого CSV с отзывами Steam;
- формирование сбалансированной подвыборки;
- построение карточек игр;
- построение пользовательских профилей;
- user-based offline evaluation со скрытой контрольной игрой;
- базовый метод TF-IDF;
- LLM-реранжирование через GigaChat;
- проверку JSON-ответов LLM, fallback и диагностику ошибок;
- экспорт финальных артефактов для диплома;
- локальный Streamlit-интерфейс для демонстрации результатов.

Проект не является production-системой и используется как учебный исследовательский прототип.

## Возможности

- chunked preprocessing большого исходного CSV без загрузки всего датасета в память;
- построение сбалансированной подвыборки для контролируемого эксперимента;
- генерация карточек игр на основе очищенных отзывов;
- построение пользовательских профилей;
- user-based holdout evaluation;
- получение baseline-рекомендаций с помощью TF-IDF;
- LLM-реранжирование подготовленного списка кандидатов;
- валидация структуры ответа LLM;
- fallback к baseline при ошибках провайдера или невалидном ответе;
- расчёт метрик и формирование отчётов;
- демонстрация итогов через Streamlit без повторного запуска экспериментов.

## Структура проекта

```text
src/                         исходный код пайплайна
configs/                     конфигурации экспериментов
data/raw/                    локальный исходный датасет Steam Reviews
data/processed/              локальные обработанные файлы
data/results/                рабочие результаты и служебные отчёты
reports/final_thesis_artifacts/  финальные артефакты для защиты
experiments/                 архивы прогонов экспериментов
app.py                       Streamlit-интерфейс
main.py                      CLI пайплайна
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Для `fish`:

```bash
source .venv/bin/activate.fish
```

## Данные

- Исходный CSV Steam Reviews должен находиться по пути `data/raw/steam_reviews.csv`.
- Сырой датасет не хранится в публичном репозитории.
- Сбалансированная подвыборка формируется отдельным шагом предобработки.
- Файл `data/processed/reviews_clean_balanced_subset.csv` является generated-артефактом и обычно слишком велик для обычного Git-репозитория, поэтому не должен считаться обязательной частью публичной поставки.

## Настройка `.env`

1. Скопируйте шаблон:

```bash
cp .env.example .env
```

2. При необходимости задайте:

- `ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv`, если используются user-based шаги на balanced subset;
- учётные данные GigaChat только если планируется реальный LLM-запуск;
- `USE_PILOT_SPLITS=true`, если нужен пилотный режим со сплитами малого размера.

3. Никогда не публикуйте `.env`.

## Основные команды

Проверка окружения:

```bash
./.venv/bin/python main.py --step check_env
```

Проверка схемы исходного CSV:

```bash
./.venv/bin/python main.py --step schema_check
```

Построение сбалансированной подвыборки:

```bash
./.venv/bin/python main.py --step preprocess_balanced_subset
```

Построение пользовательских профилей:

```bash
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step build_user_profiles
```

Построение pilot-сплитов:

```bash
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step build_user_splits_pilot
```

Запуск baseline:

```bash
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step user_baseline --config configs/experiment_config.balanced_subset_llm_10_gigachat.json
```

Dry-run LLM-промптов без реальных рекомендаций:

```bash
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step user_llm_dry_run --config configs/experiment_config.balanced_subset_llm_10_gigachat.json
```

Реальный LLM pilot:

```bash
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step run_experiment --config configs/experiment_config.balanced_subset_llm_10_gigachat.json
```

Этот шаг требует:

- корректных учётных данных GigaChat;
- сетевого доступа;
- готовности к реальным API-вызовам.

Для демонстрации и проверки репозитория этот запуск не обязателен.

## Демонстрационный интерфейс

```bash
./.venv/bin/python -m streamlit run app.py
```

Интерфейс:

- читает сохранённые финальные артефакты из `reports/final_thesis_artifacts/`;
- не перезапускает эксперименты;
- не выполняет новые запросы к языковой модели;
- показывает предупреждения, если часть артефактов отсутствует.

## Результаты

Актуальный финальный пилотный эксперимент использует:

- balanced subset;
- 791 254 очищенных отзывов;
- 200 игр;
- 712 292 пользователей в исходной processed-выгрузке;
- 1 813 пригодных пользователей для user-based оценки;
- 10-user GigaChat pilot.

Сводные метрики baseline и LLM находятся в:

- `reports/final_thesis_artifacts/user_llm_metrics_summary.csv`
- `reports/final_thesis_artifacts/user_rank_comparison.csv`
- `reports/final_thesis_artifacts/user_llm_reranking_summary.json`

Важно: LLM-результат является пилотным и не должен трактоваться как статистически окончательный вывод.

## Ограничения

- используется контролируемая подвыборка, а не production-поток на полном датасете;
- ответы LLM могут быть нестабильными;
- часть случаев обрабатывается через fallback;
- качество зависит от внешнего API-провайдера;
- offline evaluation не заменяет пользовательское онлайн-тестирование.

## Лицензия / учебное назначение

Репозиторий подготовлен как учебный исследовательский прототип для ВКР и демонстрации результатов эксперимента.
