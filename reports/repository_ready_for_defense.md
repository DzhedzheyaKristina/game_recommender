# Репозиторий готов к защите

Дата: 2026-06-23

## Назначение проекта

Учебный исследовательский прототип ВКР:
рекомендательная система игр по отзывам Steam с TF-IDF baseline, user-based evaluation и LLM-реранжированием через GigaChat.

## Сохранённые ключевые файлы

- `main.py`
- `app.py`
- `src/*.py`
- `requirements.txt`
- `.env.example`
- `configs/*.json`
- `README.md`
- `reports/final_thesis_artifacts/*`
- выбранный архив `experiments/2026-06-06_183742_steam_reviews_balanced_subset_llm_10_gigachat`

## Удалённые временные файлы

- `test_gigachat_ssl.py` — одноразовый локальный SSL-скрипт с персональным путём к сертификату

## Статус подготовки

- README переписан на русский: `да`
- Streamlit UI русифицирован: `да`
- Секреты защищены `.gitignore`: `да`
- Сырые данные исключены из Git: `да`
- Финальные thesis-артефакты присутствуют: `да`
- Реальные LLM completion-вызовы во время cleanup: `нет`

## Проверочные команды

1. `python -m py_compile main.py app.py src/*.py`
   - статус: `пройдено`

2. `./.venv/bin/python main.py --step demo_info`
   - статус: `пройдено`

3. `ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step llm_pilot_readiness --config configs/experiment_config.balanced_subset_llm_10_gigachat.json`
   - статус: `пройдено`

4. `ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step user_llm_dry_run --config configs/experiment_config.balanced_subset_llm_10_gigachat.json`
   - статус: `пройдено`

5. `./.venv/bin/python -m streamlit run app.py`
   - статус: `пройдено`
   - примечание: для проверки старта сервера потребовался запуск вне sandbox, потому что песочница запрещала открытие локального сокета

## Что важно сделать перед публикацией на GitHub

- проверить `git status` и осознанно добавить только нужные tracked-файлы;
- не публиковать `.env`, сертификаты и сырой `data/raw/steam_reviews.csv`;
- не публиковать крупные generated CSV из `data/processed/`, если они не нужны отдельно;
- решить, нужно ли коммитить локальные небольшие артефакты из `data/processed/` и `data/results/`, которые больше не игнорируются;
- при необходимости добавить краткое описание темы ВКР и лицензии на уровне репозитория;
- не перезапускать `run_experiment`, если цель только публикация и защита уже собранных результатов.
