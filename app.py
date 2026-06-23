from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_ARTIFACT_DIR = PROJECT_ROOT / "reports" / "final_thesis_artifacts"

REQUIRED_ARTIFACTS = [
    "final_experiment_summary.md",
    "experiment_manifest.json",
    "user_llm_reranking_summary.json",
    "user_llm_metrics_summary.csv",
    "user_rank_comparison.csv",
    "user_llm_explanation_examples.md",
    "balanced_subset_methodology_note.md",
]

METHOD_LABELS = {
    "user_tfidf_baseline": "Базовый метод TF-IDF",
    "user_llm_reranker": "LLM-реранжирование",
}

METRIC_LABELS = {
    "mean_hit_rate_at_5": "Попадание в топ-5",
    "mean_hit_rate_at_10": "Попадание в топ-10",
    "mean_mrr": "Средняя обратная позиция",
    "mean_ndcg_at_10": "Качество порядка в топ-10",
    "evaluated_profiles": "Оценённых профилей",
    "skipped_profiles": "Пропущенных профилей",
    "method": "Метод",
    "llm_mode": "Режим LLM",
}

STATUS_LABELS = {
    "ok": "Успешно",
    "fallback_to_baseline": "Fallback к baseline",
    "partial_fallback_to_baseline": "Частичный fallback",
    "completed_with_warnings": "Завершено с предупреждениями",
    "completed": "Завершено",
    "unknown": "Неизвестно",
}

LIMITATION_TRANSLATIONS = {
    "This is a controlled pilot on a small subset of user-based evaluation splits.": "Это контролируемый пилот на малой подвыборке пользовательских evaluation-сплитов.",
    "The results should not be presented as a full-experiment conclusion.": "Результаты не следует представлять как итог полного эксперимента.",
    "This is a real LLM pilot, but still limited to a small controlled subset.": "Это реальный LLM-пилот, но он всё ещё ограничен небольшой контролируемой подвыборкой.",
}


def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def artifact_status() -> dict[str, bool]:
    return {name: (FINAL_ARTIFACT_DIR / name).exists() for name in REQUIRED_ARTIFACTS}


def missing_artifacts() -> list[str]:
    return [name for name, exists in artifact_status().items() if not exists]


def parse_bullet_value(text: str | None, *labels: str) -> str | None:
    if not text:
        return None
    for label in labels:
        pattern = re.compile(rf"^-\s*{re.escape(label)}:\s*(.+)$", re.MULTILINE)
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def parse_int_value(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        return int(float(str(value).replace(" ", "")))
    except Exception:
        return default


def localize_method(method: object) -> str:
    return METHOD_LABELS.get(str(method), str(method))


def localize_status(status: object) -> str:
    return STATUS_LABELS.get(str(status), str(status))


def localize_split_mode(manifest: dict[str, object] | None) -> str:
    if not manifest:
        return "Неизвестно"
    split_path = str(manifest.get("active_user_splits_path", "") or "")
    if "pilot" in Path(split_path).name:
        return "Пилотный"
    if split_path:
        return "Основной"
    return "Неизвестно"


def localize_dataset_mode(dataset_mode: object) -> str:
    mapping = {
        "balanced_subset": "Сбалансированная подвыборка",
        "full": "Полный датасет",
        "subset": "Обычная подвыборка",
        "debug": "Отладочный режим",
        "unknown": "Неизвестно",
    }
    return mapping.get(str(dataset_mode), str(dataset_mode))


def localize_provider(provider: object) -> str:
    mapping = {
        "gigachat": "GigaChat",
        "openai": "OpenAI",
        "openrouter": "OpenRouter",
        "mock": "Mock",
        "unknown": "Неизвестно",
    }
    return mapping.get(str(provider).lower(), str(provider))


def localize_bool(value: object) -> str:
    return "Да" if bool(value) else "Нет"


def localize_interpretation(text: object) -> str:
    raw = str(text or "n/a")
    if raw == "LLM preserved the best holdout rank.":
        return "LLM сохранила лучшую позицию скрытой контрольной игры."

    worsened_match = re.match(r"LLM worsened the best holdout rank by (\d+) positions\.", raw)
    if worsened_match:
        delta = worsened_match.group(1)
        return f"LLM ухудшила лучшую позицию скрытой контрольной игры на {delta} поз."

    improved_match = re.match(r"LLM improved the best holdout rank by (\d+) positions\.", raw)
    if improved_match:
        delta = improved_match.group(1)
        return f"LLM улучшила лучшую позицию скрытой контрольной игры на {delta} поз."

    return raw


def localize_limitations(limitations: list[object]) -> list[str]:
    if not limitations:
        return []
    result: list[str] = []
    for item in limitations:
        text = str(item)
        result.append(LIMITATION_TRANSLATIONS.get(text, text))
    return result


def localize_explanations_markdown(text: str | None) -> str | None:
    if not text:
        return None

    replacements = {
        "# User LLM Explanation Examples": "# Примеры объяснений LLM",
        "These examples are for qualitative inspection and are not a factuality audit.": "Примеры приведены для качественного анализа и не являются проверкой фактической точности.",
        "These examples come from a controlled pilot subset.": "Все примеры получены в рамках контролируемого пилотного эксперимента.",
        "- Profile summary:": "- Краткий профиль:",
        "- Recommended game:": "- Рекомендованная игра:",
        "- LLM explanation:": "- Объяснение LLM:",
        "- Matched preferences:": "- Совпавшие предпочтения:",
        "- Possible risks:": "- Возможные риски:",
        "- Holdout in top-k:": "- Скрытая контрольная игра в топ-k:",
        "Fallback to baseline because LLM response failed validation.": "Использован fallback к baseline, потому что ответ LLM не прошёл валидацию.",
        "none": "нет",
        "True": "Да",
        "False": "Нет",
    }

    localized = text
    for source, target in replacements.items():
        localized = localized.replace(source, target)
    return localized


def localize_methodology_markdown(text: str | None) -> str | None:
    if not text:
        return None

    replacements = {
        "# Balanced Subset Methodology Note": "### Методологическая заметка по balanced subset",
        "The full Steam Reviews export is too large for a single-memory preprocessing pass in a lightweight bachelor-thesis prototype, so the project uses a chunked pipeline and a balanced subset for the controlled user-based baseline experiment.": "Полный экспорт Steam Reviews слишком велик для однопроходной обработки в памяти в рамках лёгкого исследовательского прототипа, поэтому в проекте используется потоковая предобработка и сбалансированная подвыборка.",
        "A simple first-N subset was not sufficient because the raw CSV is ordered by app_id, which concentrates early rows on only a few games and produces an overly narrow recommendation benchmark.": "Простая подвыборка первых строк оказалась непригодной: исходный CSV упорядочен по `app_id`, из-за чего ранние записи концентрируются на небольшом числе игр и искажают рекомендательный бенчмарк.",
        "The balanced subset was therefore constructed in two passes over the full raw dataset: the first pass collected per-game review statistics, and the second pass selected the most review-rich games that satisfied the minimum review thresholds.": "Сбалансированная подвыборка была собрана в два прохода по полному датасету: сначала считалась статистика по играм, затем отбирались наиболее насыщенные отзывами игры, удовлетворяющие минимальным порогам.",
        "The resulting subset contains 200 games and 791254 processed reviews, while preserving user_id so that user-based offline evaluation remains possible.": "Итоговая подвыборка содержит 200 игр и 791254 обработанных отзывов и при этом сохраняет `user_id`, что делает возможной user-based offline evaluation.",
        "The selected subset is appropriate for a controlled thesis workflow, but the resulting metrics must be reported as balanced-subset results rather than full-dataset results.": "Эта подвыборка подходит для контролируемого дипломного эксперимента, но итоговые метрики должны интерпретироваться именно как результаты на balanced subset, а не на полном датасете.",
        "The same chunked preprocessing pipeline can later be run on the full 21.7 million-row dataset if the thesis work is extended beyond the balanced subset.": "При дальнейшем развитии проекта тот же пайплайн потоковой предобработки может быть применён и к полному датасету размером 21.7 млн строк.",
    }

    localized = text
    for source, target in replacements.items():
        localized = localized.replace(source, target)
    return localized


def build_manifest_frame(
    manifest: dict[str, object],
    unique_users: int,
    eligible_users: int,
) -> pd.DataFrame:
    rows = [
        ("Статус", localize_status(manifest.get("status", "unknown"))),
        ("Режим данных", localize_dataset_mode(manifest.get("active_dataset_mode", "unknown"))),
        ("Режим сплитов", localize_split_mode(manifest)),
        ("Провайдер LLM", localize_provider(manifest.get("llm_provider", "unknown"))),
        ("Язык ответов", str(manifest.get("llm_response_language", "unknown") or "unknown")),
        ("Обработанных отзывов", parse_int_value(manifest.get("cleaned_review_count", 0))),
        ("Игр", parse_int_value(manifest.get("game_card_count", 0))),
        ("Уникальных пользователей", unique_users),
        ("Пригодных пользователей", eligible_users),
        ("Пользователей в пилоте LLM", parse_int_value(manifest.get("selected_llm_users", 0))),
        ("Попыток запроса токена", parse_int_value(manifest.get("token_requests_attempted", 0))),
        ("Попыток completion-запроса", parse_int_value(manifest.get("completion_requests_attempted", 0))),
    ]
    return pd.DataFrame(rows, columns=["Параметр", "Значение"])


def build_metrics_display_frame(metrics_df: pd.DataFrame) -> pd.DataFrame:
    display_df = metrics_df.copy()
    if "method" in display_df.columns:
        display_df["method"] = display_df["method"].map(localize_method)
    return display_df.rename(columns=METRIC_LABELS)


def build_rank_display_frame(rank_df: pd.DataFrame) -> pd.DataFrame:
    display_df = rank_df.copy()
    if "llm_status" in display_df.columns:
        display_df["llm_status"] = display_df["llm_status"].map(localize_status)
    if "interpretation" in display_df.columns:
        display_df["interpretation"] = display_df["interpretation"].map(localize_interpretation)
    rename_map = {
        "masked_user_id": "Пользователь",
        "baseline_best_holdout_rank": "Лучшая позиция holdout у baseline",
        "llm_best_holdout_rank": "Лучшая позиция holdout у LLM",
        "llm_mode": "Режим LLM",
        "rank_delta": "Изменение позиции",
        "holdout_in_candidate_pool": "Holdout в candidate pool",
        "holdout_in_baseline_top_k": "Holdout в baseline top-k",
        "holdout_in_llm_top_k": "Holdout в LLM top-k",
        "candidate_pool_size": "Размер candidate pool",
        "baseline_top_k_size": "Размер baseline top-k",
        "llm_top_k_size": "Размер LLM top-k",
        "baseline_hit_at_5": "Baseline hit@5",
        "llm_hit_at_5": "LLM hit@5",
        "baseline_hit_at_10": "Baseline hit@10",
        "llm_hit_at_10": "LLM hit@10",
        "llm_status": "Статус LLM",
        "interpretation": "Интерпретация",
    }
    return display_df.rename(columns=rename_map)


def show_artifact_warnings() -> None:
    missing = missing_artifacts()
    if not missing:
        return
    st.warning("Часть финальных артефактов отсутствует. Интерфейс покажет только доступные данные.")
    st.write("Не найдены файлы:")
    for name in missing:
        st.write(f"- `{name}`")


def show_overview_tab(
    manifest: dict[str, object] | None,
    summary: dict[str, object] | None,
    summary_md: str | None,
) -> None:
    st.subheader("Обзор эксперимента")

    if not manifest:
        st.warning("Не найден `experiment_manifest.json` в `reports/final_thesis_artifacts/`.")
        return

    unique_users = parse_int_value(
        parse_bullet_value(summary_md, "Unique users", "Уникальные пользователи"),
        default=0,
    )
    eligible_users = parse_int_value(
        parse_bullet_value(summary_md, "Eligible users", "Пригодные пользователи"),
        default=0,
    )

    cols = st.columns(4)
    cols[0].metric("Статус", localize_status(manifest.get("status", "unknown")))
    cols[1].metric("Режим данных", localize_dataset_mode(manifest.get("active_dataset_mode", "unknown")))
    cols[2].metric("Режим сплитов", localize_split_mode(manifest))
    cols[3].metric("Провайдер LLM", localize_provider(manifest.get("llm_provider", "unknown")))

    metric_cols = st.columns(4)
    metric_cols[0].metric("Обработанных отзывов", f"{parse_int_value(manifest.get('cleaned_review_count', 0)):,}".replace(",", " "))
    metric_cols[1].metric("Игр", f"{parse_int_value(manifest.get('game_card_count', 0)):,}".replace(",", " "))
    metric_cols[2].metric("Уникальных пользователей", f"{unique_users:,}".replace(",", " "))
    metric_cols[3].metric("Пригодных пользователей", f"{eligible_users:,}".replace(",", " "))

    info_cols = st.columns(4)
    info_cols[0].metric("Пользователей в пилоте LLM", parse_int_value(manifest.get("selected_llm_users", 0)))
    info_cols[1].metric("Язык ответов", str(manifest.get("llm_response_language", "unknown") or "unknown"))
    info_cols[2].metric("Попыток запроса токена", parse_int_value(manifest.get("token_requests_attempted", 0)))
    info_cols[3].metric("Completion-запросов", parse_int_value(manifest.get("completion_requests_attempted", 0)))

    st.info(
        "Финальный демонстрационный режим использует сохранённые thesis-артефакты. "
        "Здесь не пересчитываются метрики и не запускается LLM-реранжирование."
    )

    if summary:
        st.markdown(
            f"""
**Краткая интерпретация**

- Эксперимент выполнен на balanced subset.
- Реальных API-вызовов в выбранном пилоте: {parse_int_value(summary.get('real_api_calls', 0))}.
- Пользователей с полным fallback: {parse_int_value(summary.get('fallback_users', 0))}.
- Пользователей с валидными LLM-результатами: {parse_int_value(summary.get('users_with_any_valid_llm_records', 0))}.
"""
        )

    st.markdown("### Ключевые параметры запуска")
    st.dataframe(
        build_manifest_frame(manifest, unique_users, eligible_users),
        use_container_width=True,
        hide_index=True,
    )


def show_metrics_tab(metrics_df: pd.DataFrame | None, summary: dict[str, object] | None) -> None:
    st.subheader("Метрики")
    if metrics_df is None or metrics_df.empty:
        st.warning("Не найден `user_llm_metrics_summary.csv` в `reports/final_thesis_artifacts/`.")
        return

    st.info(
        "Метрики получены на малом пилотном наборе пользователей. "
        "Сравнение baseline и LLM следует интерпретировать как пилотный результат, а не как окончательный статистический вывод."
    )

    display_df = build_metrics_display_frame(metrics_df)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    metric_columns = [
        "mean_hit_rate_at_5",
        "mean_hit_rate_at_10",
        "mean_mrr",
        "mean_ndcg_at_10",
    ]
    if {"user_tfidf_baseline", "user_llm_reranker"}.issubset(set(metrics_df["method"].astype(str))):
        chart_frame = metrics_df.set_index("method")[metric_columns].T
        chart_frame.index = [METRIC_LABELS.get(metric, metric) for metric in chart_frame.index]
        chart_frame.columns = [localize_method(method) for method in chart_frame.columns]
        st.markdown("### Сравнение baseline и LLM")
        st.bar_chart(chart_frame)

    if summary:
        st.markdown("### Диагностика пилота")
        diagnostics = pd.DataFrame(
            [
                ("Выбранных пользователей", parse_int_value(summary.get("selected_users", 0))),
                ("Пользователей с fallback", parse_int_value(summary.get("fallback_users", 0))),
                ("Пользователей с частичным fallback", parse_int_value(summary.get("partial_fallback_users", 0))),
                ("Пользователей с ошибками схемы", parse_int_value(summary.get("users_with_schema_errors", 0))),
                ("Провайдер прошёл preflight", localize_bool(summary.get("provider_preflight_ok", False))),
            ],
            columns=["Показатель", "Значение"],
        )
        st.dataframe(diagnostics, use_container_width=True, hide_index=True)


def show_rank_comparison_tab(rank_df: pd.DataFrame | None) -> None:
    st.subheader("Сравнение рангов")
    if rank_df is None or rank_df.empty:
        st.warning("Не найден `user_rank_comparison.csv` в `reports/final_thesis_artifacts/`.")
        return

    st.dataframe(build_rank_display_frame(rank_df), use_container_width=True, hide_index=True)

    if "masked_user_id" not in rank_df.columns:
        st.info("В таблице нет `masked_user_id`, поэтому выбор отдельного пользователя недоступен.")
        return

    masked_ids = rank_df["masked_user_id"].astype(str).tolist()
    selected_masked_id = st.selectbox("Выберите пользователя", masked_ids)
    selected_row = rank_df[rank_df["masked_user_id"].astype(str) == selected_masked_id].iloc[0]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Лучшая позиция holdout у baseline", str(selected_row.get("baseline_best_holdout_rank", "n/a")))
    metric_cols[1].metric("Лучшая позиция holdout у LLM", str(selected_row.get("llm_best_holdout_rank", "n/a")))
    metric_cols[2].metric("Изменение позиции", str(selected_row.get("rank_delta", "n/a")))
    metric_cols[3].metric("Статус LLM", localize_status(selected_row.get("llm_status", "n/a")))

    st.markdown(f"**Интерпретация:** {localize_interpretation(selected_row.get('interpretation', 'n/a'))}")


def show_explanations_tab(explanations_md: str | None) -> None:
    st.subheader("Примеры объяснений")
    st.info("Ниже показаны сохранённые примеры объяснений рекомендаций из финального пилотного эксперимента.")
    localized_text = localize_explanations_markdown(explanations_md)
    if localized_text:
        st.markdown(localized_text)
    else:
        st.warning("Не найден `user_llm_explanation_examples.md` в `reports/final_thesis_artifacts/`.")


def show_limitations_tab(methodology_md: str | None, summary: dict[str, object] | None) -> None:
    st.subheader("Ограничения")
    st.warning("Это демонстрация контролируемого пилотного эксперимента, а не production-система и не итог на полном датасете.")

    st.markdown(
        """
- Используется balanced subset, а не полный Steam Reviews датасет.
- Пилотный запуск охватывает только небольшой набор пользователей.
- Ответы LLM могут быть нестабильными и требовать fallback.
- Качество зависит от доступности и корректности внешнего API-провайдера.
- Offline evaluation не заменяет онлайн-валидацию на реальных пользователях.
"""
    )

    if summary:
        pilot_limitations = localize_limitations(list(summary.get("pilot_limitations", [])))
        if pilot_limitations:
            st.markdown("### Ограничения пилота")
            for item in pilot_limitations:
                st.markdown(f"- {item}")

    localized_methodology = localize_methodology_markdown(methodology_md)
    if localized_methodology:
        st.markdown(localized_methodology)
    else:
        st.info("Методологическая заметка о balanced subset недоступна.")


def main() -> None:
    st.set_page_config(
        page_title="Демонстрация рекомендательной системы для игр Steam",
        page_icon="🎮",
        layout="wide",
    )

    st.title("Демонстрация рекомендательной системы для игр Steam")
    st.subheader("TF-IDF baseline + LLM-реранжирование с объяснениями")
    st.info("Интерфейс отображает сохранённые результаты эксперимента и не выполняет новые запросы к языковой модели.")

    show_artifact_warnings()

    manifest = load_json(FINAL_ARTIFACT_DIR / "experiment_manifest.json")
    summary = load_json(FINAL_ARTIFACT_DIR / "user_llm_reranking_summary.json")
    metrics_df = load_csv(FINAL_ARTIFACT_DIR / "user_llm_metrics_summary.csv")
    rank_df = load_csv(FINAL_ARTIFACT_DIR / "user_rank_comparison.csv")
    summary_md = load_text(FINAL_ARTIFACT_DIR / "final_experiment_summary.md")
    explanations_md = load_text(FINAL_ARTIFACT_DIR / "user_llm_explanation_examples.md")
    methodology_md = load_text(FINAL_ARTIFACT_DIR / "balanced_subset_methodology_note.md")

    tabs = st.tabs(
        [
            "Обзор эксперимента",
            "Метрики",
            "Сравнение рангов",
            "Примеры объяснений",
            "Ограничения",
        ]
    )

    with tabs[0]:
        show_overview_tab(manifest, summary, summary_md)
    with tabs[1]:
        show_metrics_tab(metrics_df, summary)
    with tabs[2]:
        show_rank_comparison_tab(rank_df)
    with tabs[3]:
        show_explanations_tab(explanations_md)
    with tabs[4]:
        show_limitations_tab(methodology_md, summary)


if __name__ == "__main__":
    main()
