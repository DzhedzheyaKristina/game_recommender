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


def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    return path.read_text(encoding="utf-8")


def artifact_status() -> dict[str, bool]:
    return {name: (FINAL_ARTIFACT_DIR / name).exists() for name in REQUIRED_ARTIFACTS}


def missing_artifacts() -> list[str]:
    return [name for name, exists in artifact_status().items() if not exists]


def parse_bullet_value(text: str, label: str) -> str | None:
    pattern = re.compile(rf"^-\s*{re.escape(label)}:\s*(.+)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


def infer_split_mode(manifest: dict[str, object] | None) -> str:
    if not manifest:
        return "unknown"
    split_path = str(manifest.get("active_user_splits_path", "") or "")
    if "pilot" in Path(split_path).name:
        return "pilot"
    if split_path:
        return "main"
    return "unknown"


def show_artifact_warnings() -> None:
    missing = missing_artifacts()
    if missing:
        st.warning(
            "Отсутствуют некоторые финальные артефакты. Интерфейс покажет доступные данные и предупредит о пробелах."
        )
        st.write("Не найдены файлы:")
        for name in missing:
            st.write(f"- `{name}`")


def show_overview_tab(manifest: dict[str, object] | None, summary: dict[str, object] | None, summary_md: str | None) -> None:
    st.subheader("Обзор эксперимента")

    if not manifest:
        st.warning("Не найден experiment_manifest.json в reports/final_thesis_artifacts/.")
        return

    summary_values = summary or {}
    split_mode = infer_split_mode(manifest)
    dataset_mode = str(manifest.get("active_dataset_mode", "unknown") or "unknown")
    status = str(manifest.get("status", "unknown") or "unknown")

    cols = st.columns(4)
    cols[0].metric("Статус", status)
    cols[1].metric("Режим данных", dataset_mode)
    cols[2].metric("Режим сплитов", split_mode)
    cols[3].metric("LLM provider", str(manifest.get("llm_provider", "unknown") or "unknown"))

    metric_cols = st.columns(4)
    metric_cols[0].metric("Processed reviews", f"{int(manifest.get('cleaned_review_count', 0) or 0):,}".replace(",", " "))
    metric_cols[1].metric("Games", f"{int(manifest.get('game_card_count', 0) or 0):,}".replace(",", " "))
    metric_cols[2].metric("Unique users", f"{int(summary_values.get('unique_users', parse_bullet_value(summary_md or '', 'Unique users') or 0) or 0):,}".replace(",", " "))
    metric_cols[3].metric("Eligible users", f"{int(summary_values.get('eligible_users', parse_bullet_value(summary_md or '', 'Eligible users') or 0) or 0):,}".replace(",", " "))

    info_cols = st.columns(4)
    info_cols[0].metric("Selected LLM users", int(manifest.get("selected_llm_users", 0) or 0))
    info_cols[1].metric("Response language", str(manifest.get("llm_response_language", "unknown") or "unknown"))
    info_cols[2].metric("Token requests", int(manifest.get("token_requests_attempted", 0) or 0))
    info_cols[3].metric("Completion requests", int(manifest.get("completion_requests_attempted", 0) or 0))

    st.caption("Интерфейс читает только финальные архивированные артефакты и не запускает эксперименты заново.")
    if summary_md:
        st.markdown(summary_md)
    else:
        st.info("Файл final_experiment_summary.md не найден.")

    st.markdown("### Ключевые поля манифеста")
    manifest_frame = pd.DataFrame(
        [
            {
                "dataset_mode": dataset_mode,
                "processed_reviews": manifest.get("cleaned_review_count", 0),
                "game_count": manifest.get("game_card_count", 0),
                "unique_users": parse_bullet_value(summary_md or "", "Unique users") or "n/a",
                "eligible_users": parse_bullet_value(summary_md or "", "Eligible users") or "n/a",
                "active_split_mode": split_mode,
                "selected_llm_users": manifest.get("selected_llm_users", 0),
                "llm_provider": manifest.get("llm_provider", "unknown"),
                "response_language": manifest.get("llm_response_language", "unknown"),
                "token_requests": manifest.get("token_requests_attempted", 0),
                "completion_requests": manifest.get("completion_requests_attempted", 0),
                "status": status,
            }
        ]
    )
    st.dataframe(manifest_frame, use_container_width=True, hide_index=True)


def show_metrics_tab(metrics_df: pd.DataFrame | None, summary: dict[str, object] | None) -> None:
    st.subheader("Метрики")
    if metrics_df is None or metrics_df.empty:
        st.warning("Не найден user_llm_metrics_summary.csv в reports/final_thesis_artifacts/.")
        return

    st.info(
        "Метрики рассчитаны на малом пилотном наборе. Baseline и LLM могут иметь разное число evaluated_profiles, "
        "поэтому результаты следует интерпретировать осторожно."
    )

    if {"evaluated_profiles", "skipped_profiles"}.issubset(metrics_df.columns):
        profile_counts = metrics_df[["method", "evaluated_profiles", "skipped_profiles"]].copy()
        st.markdown("### Объем оценки")
        st.dataframe(profile_counts, use_container_width=True, hide_index=True)

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    metric_columns = ["mean_hit_rate_at_5", "mean_hit_rate_at_10", "mean_mrr", "mean_ndcg_at_10"]
    if {"user_tfidf_baseline", "user_llm_reranker"}.issubset(set(metrics_df["method"].astype(str))):
        chart_frame = metrics_df.set_index("method")[metric_columns].T
        st.markdown("### Сравнение базовой модели и LLM")
        st.bar_chart(chart_frame)

    if summary:
        st.markdown("### Итоговый итог пилота")
        st.json(
            {
                "baseline_metrics": summary.get("baseline_metrics", {}),
                "llm_metrics": summary.get("llm_metrics", {}),
                "selected_users": summary.get("selected_users", 0),
                "fallback_users": summary.get("fallback_users", 0),
                "failed_users": summary.get("failed_users", 0),
                "provider_preflight_ok": summary.get("provider_preflight_ok", False),
            }
        )


def show_rank_comparison_tab(rank_df: pd.DataFrame | None) -> None:
    st.subheader("Сравнение рангов")
    if rank_df is None or rank_df.empty:
        st.warning("Не найден user_rank_comparison.csv в reports/final_thesis_artifacts/.")
        return

    display_df = rank_df.copy()
    if "user_id" in display_df.columns:
        display_df = display_df.drop(columns=["user_id"])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if "masked_user_id" not in rank_df.columns:
        st.info("В таблице сравнения нет masked_user_id для выбора пользователя.")
        return

    masked_ids = rank_df["masked_user_id"].astype(str).tolist()
    selected_masked_id = st.selectbox("Выберите пользователя", masked_ids)
    selected_row = rank_df[rank_df["masked_user_id"].astype(str) == selected_masked_id].iloc[0]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Baseline best holdout rank", str(selected_row.get("baseline_best_holdout_rank", "n/a")))
    metric_cols[1].metric("LLM best holdout rank", str(selected_row.get("llm_best_holdout_rank", "n/a")))
    metric_cols[2].metric("Rank delta", str(selected_row.get("rank_delta", "n/a")))
    metric_cols[3].metric("LLM status", str(selected_row.get("llm_status", "n/a")))

    st.markdown(f"**Interpretation:** {selected_row.get('interpretation', 'n/a')}")


def show_explanations_tab(explanations_md: str | None) -> None:
    st.subheader("Примеры объяснений")
    st.info("Примеры предназначены для качественной оценки объяснимости рекомендаций.")
    if explanations_md:
        st.markdown(explanations_md)
    else:
        st.warning("Не найден user_llm_explanation_examples.md в reports/final_thesis_artifacts/.")


def show_limitations_tab(methodology_md: str | None, summary_md: str | None, summary: dict[str, object] | None) -> None:
    st.subheader("Ограничения")
    st.warning(
        "Это контролируемый пилот на balanced subset, а не полный датасет и не статистически значимый итоговый эксперимент."
    )
    st.markdown(
        """
- Controlled pilot only.
- Balanced subset, not the full Steam Reviews dataset.
- Limited number of users.
- Depends on external LLM API availability.
- Fallback cases are present and should be reported explicitly.
- The results are not statistically significant.
"""
    )
    if summary_md:
        st.markdown(summary_md)
    if methodology_md:
        st.markdown(methodology_md)
    if summary:
        st.markdown("### Дополнительные сведения")
        st.json(
            {
                "status": summary.get("status", "unknown"),
                "llm_mode": summary.get("llm_mode", "unknown"),
                "llm_provider": summary.get("llm_provider", "unknown"),
                "selected_users": summary.get("selected_users", 0),
                "fallback_users": summary.get("fallback_users", 0),
                "failed_users": summary.get("failed_users", 0),
            }
        )


def main() -> None:
    st.set_page_config(
        page_title="Демонстрация рекомендательной системы для игр Steam",
        page_icon="🎮",
        layout="wide",
    )

    st.title("Демонстрация рекомендательной системы для игр Steam")
    st.subheader("TF-IDF baseline + LLM-реранжирование с объяснениями")

    show_artifact_warnings()

    manifest = load_json(FINAL_ARTIFACT_DIR / "experiment_manifest.json")
    summary = load_json(FINAL_ARTIFACT_DIR / "user_llm_reranking_summary.json")
    metrics_df = load_csv(FINAL_ARTIFACT_DIR / "user_llm_metrics_summary.csv")
    rank_df = load_csv(FINAL_ARTIFACT_DIR / "user_rank_comparison.csv")
    summary_md = load_text(FINAL_ARTIFACT_DIR / "final_experiment_summary.md")
    explanations_md = load_text(FINAL_ARTIFACT_DIR / "user_llm_explanation_examples.md")
    methodology_md = load_text(FINAL_ARTIFACT_DIR / "balanced_subset_methodology_note.md")

    tabs = st.tabs([
        "Обзор эксперимента",
        "Метрики",
        "Сравнение рангов",
        "Примеры объяснений",
        "Ограничения",
    ])

    with tabs[0]:
        show_overview_tab(manifest, summary, summary_md)
    with tabs[1]:
        show_metrics_tab(metrics_df, summary)
    with tabs[2]:
        show_rank_comparison_tab(rank_df)
    with tabs[3]:
        show_explanations_tab(explanations_md)
    with tabs[4]:
        show_limitations_tab(methodology_md, summary_md, summary)


if __name__ == "__main__":
    main()
