# Аудит очистки репозитория

Дата аудита: 2026-06-23

Статус на момент создания: аудит выполнен до удаления файлов.

## 1. Важные исходные файлы

- `main.py` — основной CLI-энтрипойнт пайплайна.
- `app.py` — локальный Streamlit-интерфейс для демонстрации финальных артефактов.
- `src/config.py` — загрузка настроек, путей и переменных окружения.
- `src/environment_tools.py` — проверки окружения и зависимостей.
- `src/data_loader.py` — загрузка и валидация CSV.
- `src/preprocessing.py` — chunked preprocessing, subset/debug/balanced subset режимы.
- `src/game_card_builder.py` — построение карточек игр.
- `src/scenario_builder.py` — сценарии и вспомогательная логика.
- `src/baseline_tfidf.py` — базовый TF-IDF метод.
- `src/llm_provider.py` — интеграции провайдеров LLM.
- `src/llm_reranker.py` — LLM-реранжирование.
- `src/user_experiments.py` — user-based профили, сплиты и базовая оценка.
- `src/user_llm.py` — user-based LLM pilot, dry-run, валидация, fallback, отчёты.
- `src/evaluation.py` — метрики и агрегаты оценки.
- `src/experiment_tools.py` — preflight, readiness, экспорт и выбор финального эксперимента.
- `src/utils.py` — служебные функции.
- `prompts/system_prompt.txt`, `prompts/reranking_prompt_template.txt` — промпты LLM.

## 2. Важные конфиги и служебные файлы

- `requirements.txt` — зависимости.
- `.env.example` — шаблон переменных окружения.
- `.gitignore` — политика публикации данных и секретов.
- `Makefile` — вспомогательные команды.
- `configs/experiment_config.balanced_subset_baseline.json`
- `configs/experiment_config.balanced_subset_llm_10_gigachat.json`
- `configs/experiment_config.balanced_subset_llm_mock.json`
- `configs/experiment_config.balanced_subset_llm_pilot.json`
- `configs/experiment_config.balanced_subset_llm_tiny.json`
- `configs/experiment_config.balanced_subset_llm_tiny_gigachat.json`
- `configs/experiment_config.debug.json` — debug-конфиг, пока не удалять: используется CLI и упоминается в README.
- `configs/experiment_config.example.json`
- `scripts/run_preflight.sh`, `scripts/run_analysis.sh`, `scripts/run_full_pipeline.sh`, `scripts/smoke_test.sh`

## 3. Важные артефакты данных и результатов, которые нужно сохранить

Обязательные локальные processed/result артефакты:

- `data/processed/game_cards.jsonl`
- `data/processed/user_profiles.jsonl`
- `data/processed/user_evaluation_splits_pilot.jsonl`
- `data/processed/reviews_clean_balanced_subset.csv`
- `data/results/user_llm_metrics_summary.csv`
- `data/results/user_rank_comparison.csv`
- `data/results/user_llm_reranking_summary.json`
- `data/results/user_llm_validation_summary.json`
- `data/results/experiment_manifest.json`

Финальные артефакты для защиты, уже присутствуют и должны быть сохранены:

- `reports/final_thesis_artifacts/final_experiment_summary.md`
- `reports/final_thesis_artifacts/experiment_manifest.json`
- `reports/final_thesis_artifacts/user_llm_reranking_summary.json`
- `reports/final_thesis_artifacts/user_llm_metrics_summary.csv`
- `reports/final_thesis_artifacts/user_rank_comparison.csv`
- `reports/final_thesis_artifacts/user_llm_explanation_examples.md`
- `reports/final_thesis_artifacts/user_llm_pilot_summary.md`
- `reports/final_thesis_artifacts/thesis_balanced_subset_dataset_table.md`
- `reports/final_thesis_artifacts/user_thesis_metrics_table.md`
- `reports/final_thesis_artifacts/balanced_subset_methodology_note.md`

Важные отчёты для диплома и демонстрации:

- `reports/final_experiment_selection.md`
- `reports/balanced_subset_methodology_note.md`
- `reports/thesis_balanced_subset_dataset_table.md`
- `reports/user_thesis_metrics_table.md`
- `reports/user_llm_pilot_summary.md`
- `reports/user_rank_comparison.md`
- `reports/user_llm_explanation_examples.md`
- `reports/user_evaluation_split_pilot_summary.md`

Выбранный финальный архив эксперимента:

- `experiments/2026-06-06_183742_steam_reviews_balanced_subset_llm_10_gigachat`

## 4. Крупные локальные файлы и публикационные риски

Обнаружены крупные локальные файлы:

- `data/raw/steam_reviews.csv` — 7.7 ГБ, не должен публиковаться.
- `data/processed/reviews_clean.csv` — 577 МБ.
- `data/processed/reviews_clean_balanced_subset.csv` — 554 МБ.
- `data/processed/reviews_clean_subset.csv` — 792 МБ.
- `data/processed/reviews_clean_debug.csv` — 247 МБ.
- `data/processed/game_cards.jsonl` — 2.6 МБ.
- `data/processed/user_profiles.jsonl` — 4.4 МБ.

Вывод:

- сырой датасет однозначно не должен попадать в публичный Git;
- processed CSV слишком крупные для обычного Git-репозитория и должны считаться локально сгенерированными артефактами;
- финальные thesis-артефакты в `reports/final_thesis_artifacts/` компактные и пригодны для коммита.

## 5. Кандидаты на временные/debug-файлы

Ничего не удалено на этапе аудита. Ниже только кандидаты:

- `test_gigachat_ssl.py`
  - отдельный одноразовый SSL-скрипт;
  - не импортируется и не вызывается из `main.py`, `app.py`, `src/`, `scripts/`, `README.md`;
  - содержит локальный путь `/home/kristina/certs-gigachat/gigachat_ca_bundle.pem`;
  - кандидат на удаление из публичного репозитория.
- `data/results/preprocessing_debug_summary.json`
- `data/results/preprocessing_debug_summary.md`
- `data/results/user_llm_prompt_preview.json`
- `data/results/user_llm_prompt_preview.md`
- `data/results/user_llm_prompt_preview_10_gigachat.json`
- `data/results/user_llm_prompt_preview_10_gigachat.md`
- `data/results/user_llm_prompt_preview_tiny.json`
- `data/results/user_llm_prompt_preview_tiny.md`
- `data/results/user_llm_mock_validation_summary.json`
  - не удалять автоматически: нужны для диагностики и mock-проверок, но не должны засорять публичный индекс при регулярной работе.

## 6. Кандидаты на неиспользуемые или спорные файлы

- `test_gigachat_ssl.py` — наиболее вероятный неиспользуемый файл.
- `venv/`, `.venv/`, `__pycache__/`, `src/__pycache__/` — локальные окружения и кэш, не должны коммититься.
- `configs/experiment_config.debug.json` — debug-файл, но пока остаётся: поддерживается CLI и документирован.

## 7. Файлы, которые не должны коммититься

- `.env`
- любые реальные API-ключи и токены
- любые локальные сертификаты и CA bundle
- `data/raw/*.csv`
- `data/raw/*.zip`
- `data/raw/*.parquet`
- крупные generated CSV в `data/processed/`
- временные `*.tmp`
- локальные виртуальные окружения `venv/`, `.venv/`
- `__pycache__/`
- потенциальные debug/result кэши и prompt preview файлы
- Streamlit/cache каталоги, если будут создаваться локально

Отдельный риск:

- `test_gigachat_ssl.py` ссылается на персональный сертификатный путь и не подходит для публичного репозитория.

## 8. Английский текст в README и UI

README:

- текущий `README.md` почти полностью на английском и слишком объёмен для публичной защиты;
- содержит длинную историю запуска, отладочные детали и подробности, которые лучше сократить.

Streamlit UI (`app.py`):

- уже частично русифицирован, но есть английские человеко-ориентированные подписи:
  - `LLM provider`
  - `Processed reviews`
  - `Games`
  - `Unique users`
  - `Eligible users`
  - `Selected LLM users`
  - `Response language`
  - `Token requests`
  - `Completion requests`
  - `dataset_mode`
  - `processed_reviews`
  - `game_count`
  - `unique_users`
  - `eligible_users`
  - `active_split_mode`
  - `selected_llm_users`
  - `llm_provider`
  - `response_language`
  - `token_requests`
  - `completion_requests`
  - `status`
  - `Baseline best holdout rank`
  - `LLM best holdout rank`
  - `Rank delta`
  - `LLM status`
  - `Interpretation`
  - блок ограничений `Controlled pilot only`, `Balanced subset`, `Limited number of users`, `Depends on external LLM API availability`, `Fallback cases are present`, `The results are not statistically significant`

## 9. TODO / FIXME / debug-комментарии

Поиск по `TODO` и `FIXME`:

- явных `TODO` не найдено;
- явных `FIXME` не найдено.

Найденные debug/mock-сигналы:

- `main.py` поддерживает `preprocess_debug`.
- `src/config.py`, `src/preprocessing.py`, `src/user_experiments.py`, `src/experiment_tools.py` содержат debug-пути и debug-режимы.
- `src/user_llm.py` и `src/llm_provider.py` содержат mock/debug/SSL-ветки.

Вывод:

- это рабочая часть исследовательского прототипа, а не автоматически удаляемый мусор;
- debug/mock-поддержку удалять нельзя без отдельного доказательства неиспользуемости.

## 10. Очевидные дубликаты и потенциально устаревшие материалы

Потенциальные дубликаты отчётов:

- `reports/*` и `reports/final_thesis_artifacts/*` частично содержат однотипные документы для текущих результатов и экспортированных финальных артефактов.
- `data/results/*` и `reports/final_thesis_artifacts/*` пересекаются по смыслу, но используются по-разному:
  - `data/results/` — рабочие mutable-выгрузки;
  - `reports/final_thesis_artifacts/` — зафиксированный финальный набор для защиты.

Потенциально устаревшие архивы:

- `experiments/2026-06-03_220855_steam_reviews_debug_experiment`
- два baseline-архива от `2026-06-05`
- несколько mock-validation архивов
- много tiny GigaChat архивов от `2026-06-05` и `2026-06-06`

Важно:

- архивы экспериментов не удалять автоматически;
- сначала достаточно зафиксировать финально выбранный архив и оставить остальные как историю исследования.

## 11. Ссылки и проверки использования кандидатов

- `test_gigachat_ssl.py`
  - ссылок в `README.md`, `app.py`, `main.py`, `src/`, `scripts/`, `reports/`, `configs/` не найдено;
  - на этапе аудита классифицирован как безопасный кандидат на удаление.

- `configs/experiment_config.debug.json`
  - упоминается в `README.md`;
  - debug-шаги поддерживаются CLI;
  - пока оставить.

- mock validation отчёты и summary
  - используются логикой `src/user_llm.py` и `src/experiment_tools.py`;
  - не считать мёртвым кодом.

## 12. Промежуточный вывод перед очисткой

Безопасные направления правок:

- переписать `README.md` на русский и сократить его;
- русифицировать человеко-ориентированные строки в `app.py`;
- усилить `.gitignore` для секретов, кэшей, временных и крупных generated-файлов;
- удалить или исключить из публикации только явно одноразовые локальные файлы.

Осторожность требуется для:

- архивов `experiments/`;
- mock/debug поддержки в `src/`;
- рабочих отчётов в `data/results/`.

## 13. Итог после очистки

- Удалённые файлы:
  - `test_gigachat_ssl.py`
    - удалён как одноразовый локальный SSL-скрипт;
    - не был связан с основным CLI, Streamlit или `src/`;
    - содержал персональный путь к локальному сертификату.

- Оставлены и сохранены осознанно:
  - `configs/experiment_config.debug.json`
    - оставлен, так как поддерживается CLI и упоминается в документации.
  - mock/debug-ветки в `src/user_llm.py`, `src/llm_provider.py`, `src/preprocessing.py`
    - оставлены, так как используются для dry-run, readiness и fallback/валидации.
  - `reports/final_thesis_artifacts/*`
    - сохранены без перезаписи как финальный набор артефактов для защиты.
  - `experiments/2026-06-06_183742_steam_reviews_balanced_subset_llm_10_gigachat`
    - сохранён как выбранный архив финального пилота.
  - прочие архивы `experiments/*`
    - не удалялись автоматически, так как являются историей исследования.
