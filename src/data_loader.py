"""Load raw review data, validate schemas, and inspect dataset files."""

from __future__ import annotations

from pathlib import Path
import ast
import json
import re

import pandas as pd

from src.config import Settings
from src.utils import get_logger, read_jsonl


BASIC_MODE_REQUIRED_COLUMNS = {
    "app_id",
    "app_name",
    "review_id",
    "review",
    "recommended",
}
USER_MODE_REQUIRED_COLUMNS = BASIC_MODE_REQUIRED_COLUMNS | {"author.steamid"}
HEADER_VARIANTS = {
    "": "Index",
    "index": "Index",
    "unnamed: 0": "Index",
    "unnamed:0": "Index",
    "app_id": "app_id",
    "app_name": "app_name",
    "review_id": "review_id",
    "language": "language",
    "review": "review",
    "timestamp_created": "timestamp_created",
    "timestamp_updated": "timestamp_updated",
    "recommended": "recommended",
    "votes_helpful": "votes_helpful",
    "votes_funny": "votes_funny",
    "weighted_vote_score": "weighted_vote_score",
    "comment_count": "comment_count",
    "steam_purchase": "steam_purchase",
    "received_for_free": "received_for_free",
    "written_during_early_access": "written_during_early_access",
    "author.steamid": "author.steamid",
    "author_steamid": "author.steamid",
    "author.steam_id": "author.steamid",
    "steamid": "author.steamid",
    "user_id": "author.steamid",
    "author.num_games_owned": "author.num_games_owned",
    "author_num_games_owned": "author.num_games_owned",
    "author.num_reviews": "author.num_reviews",
    "author_num_reviews": "author.num_reviews",
    "author.playtime_forever": "author.playtime_forever",
    "author_playtime_forever": "author.playtime_forever",
    "author.playtime_last_two_weeks": "author.playtime_last_two_weeks",
    "author_playtime_last_two_weeks": "author.playtime_last_two_weeks",
    "author.playtime_at_review": "author.playtime_at_review",
    "author_playtime_at_review": "author.playtime_at_review",
    "author.last_played": "author.last_played",
    "author_last_played": "author.last_played",
}
RAW_PREVIEW_COLUMNS = [
    "app_id",
    "app_name",
    "review_id",
    "review",
    "recommended",
    "author.steamid",
]
SAMPLE_WARNING_MESSAGE = (
    "This file still looks like a sample or incomplete dataset. "
    "It is not suitable for the final user-based thesis experiment."
)
STREAM_CHUNK_SIZE = 1_000_000


def load_reviews_csv(settings: Settings) -> pd.DataFrame:
    """Load the selected Steam reviews CSV and validate its minimum schema."""

    dataset_path = select_reviews_csv_path(settings)
    report = validate_dataset_schema(settings, dataset_path)
    if report["status"] == "error":
        raise ValueError(
            "Reviews CSV is missing required columns: "
            + ", ".join(report["missing_expected_columns"])
        )

    reviews_df = pd.read_csv(dataset_path)
    resolution = build_column_resolution(list(reviews_df.columns))
    missing_basic_columns = [
        column
        for column in BASIC_MODE_REQUIRED_COLUMNS
        if column not in resolution["canonical_to_raw"]
    ]
    if missing_basic_columns:
        raise ValueError(
            "Reviews CSV is missing required basic-mode columns: "
            + ", ".join(sorted(missing_basic_columns))
        )
    return reviews_df


def select_reviews_csv_path(settings: Settings) -> Path:
    """Resolve the dataset path from config or available files in data/raw/."""

    logger = get_logger()
    configured_path = settings.reviews_csv_path
    if configured_path.exists():
        logger.info("Using dataset file: %s", configured_path)
        return configured_path

    available_csvs = sorted(settings.raw_data_dir.glob("*.csv"))
    if len(available_csvs) == 1:
        detected_path = available_csvs[0]
        logger.warning(
            "Configured dataset path %s does not exist. Using the only CSV found in data/raw/: %s",
            configured_path,
            detected_path,
        )
        settings.reviews_csv_path = detected_path
        return detected_path

    if not available_csvs:
        raise FileNotFoundError(
            f"Configured reviews CSV not found at '{configured_path}' and no CSV files were found in '{settings.raw_data_dir}'."
        )

    available_list = "\n".join(f"- {path}" for path in available_csvs)
    raise FileNotFoundError(
        "Configured reviews CSV was not found and multiple CSV files are available in data/raw/.\n"
        "Set STEAM_REVIEWS_CSV explicitly in .env or config.\n"
        f"Available CSV files:\n{available_list}"
    )


def normalize_raw_column_name(name: str) -> str:
    """Normalize known raw Steam-review column names while preserving semantics."""

    cleaned = str(name).replace("\ufeff", "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    lowered = cleaned.lower()
    return HEADER_VARIANTS.get(lowered, cleaned)


def build_column_resolution(columns: list[str]) -> dict[str, object]:
    """Build canonical/raw column mappings and diagnostic warnings."""

    raw_columns = [str(column) for column in columns]
    normalized_columns = [normalize_raw_column_name(column) for column in raw_columns]
    canonical_to_raw: dict[str, str] = {}
    duplicate_matches: dict[str, list[str]] = {}
    for raw_column, normalized_column in zip(raw_columns, normalized_columns, strict=False):
        duplicate_matches.setdefault(normalized_column, []).append(raw_column)
        canonical_to_raw.setdefault(normalized_column, raw_column)

    warnings: list[str] = []
    if raw_columns and raw_columns[0].startswith("\ufeff"):
        warnings.append("Byte order mark detected in first column name.")
    if any(column != column.strip() for column in raw_columns):
        warnings.append("Leading or trailing spaces detected in column names.")
    if len(raw_columns) == 1 and any(token in raw_columns[0] for token in [",", ";", "\t"]):
        warnings.append(
            "Only one column was detected. The file may use an unexpected separator."
        )
    if any(len(matches) > 1 for matches in duplicate_matches.values()):
        duplicated = [
            normalized
            for normalized, matches in duplicate_matches.items()
            if len(matches) > 1
        ]
        warnings.append(
            "Multiple raw columns map to the same canonical name: " + ", ".join(duplicated)
        )

    return {
        "raw_columns": raw_columns,
        "normalized_columns": normalized_columns,
        "canonical_to_raw": canonical_to_raw,
        "column_name_warnings": warnings,
        "has_exact_author_steamid": "author.steamid" in raw_columns,
        "has_any_steamid_column": any("steamid" in column.lower() for column in raw_columns),
        "has_any_author_column": any("author" in column.lower() for column in raw_columns),
    }


def run_schema_check(settings: Settings) -> dict[str, object]:
    """Locate the dataset, validate the header, and save schema reports."""

    dataset_path = select_reviews_csv_path(settings)
    report = validate_dataset_schema(settings, dataset_path)
    logger = get_logger()
    logger.info(
        "Schema check completed for %s: status=%s, canonical columns detected=%s/%s",
        dataset_path,
        report["status"],
        len(report["present_columns"]),
        len(report["expected_columns"]),
    )
    if report["missing_expected_columns"]:
        logger.warning(
            "Missing expected columns: %s",
            ", ".join(report["missing_expected_columns"]),
        )
    if report["column_name_warnings"]:
        logger.warning(
            "Column diagnostics: %s",
            " ; ".join(report["column_name_warnings"]),
        )
    return report


def validate_dataset_schema(settings: Settings, dataset_path: Path) -> dict[str, object]:
    """Validate the dataset header and persist a compact report."""

    raw_columns = read_dataset_header(dataset_path)
    resolution = build_column_resolution(raw_columns)
    present_columns = [
        column
        for column in resolution["canonical_to_raw"].keys()
        if column in settings.expected_full_schema_columns
    ]
    expected_columns = list(settings.expected_full_schema_columns)
    missing_expected_columns = [
        column for column in expected_columns if column not in present_columns
    ]
    extra_columns = [
        column
        for column in resolution["normalized_columns"]
        if column not in expected_columns
    ]
    required_for_basic_mode_present = all(
        column in resolution["canonical_to_raw"] for column in BASIC_MODE_REQUIRED_COLUMNS
    )
    required_for_user_mode_present = all(
        column in resolution["canonical_to_raw"] for column in USER_MODE_REQUIRED_COLUMNS
    )
    user_id_column_detected = "author.steamid" in resolution["canonical_to_raw"]
    status = "ok"
    if not required_for_basic_mode_present:
        status = "error"
    elif missing_expected_columns or resolution["column_name_warnings"]:
        status = "warning"

    report = {
        "dataset_path": str(dataset_path),
        "expected_columns": expected_columns,
        "present_columns": present_columns,
        "raw_columns": resolution["raw_columns"],
        "normalized_columns": resolution["normalized_columns"],
        "missing_expected_columns": missing_expected_columns,
        "extra_columns": extra_columns,
        "required_for_basic_mode_present": required_for_basic_mode_present,
        "required_for_user_mode_present": required_for_user_mode_present,
        "user_id_column_detected": user_id_column_detected,
        "matched_user_id_source_column": resolution["canonical_to_raw"].get("author.steamid", ""),
        "column_name_warnings": resolution["column_name_warnings"],
        "status": status,
    }
    settings.schema_validation_report_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    settings.schema_validation_report_markdown_path.write_text(
        build_schema_validation_markdown(report),
        encoding="utf-8",
    )
    return report


def inspect_raw_dataset(settings: Settings) -> dict[str, object]:
    """Inspect the raw CSV before preprocessing and save compact reports."""

    dataset_path = select_reviews_csv_path(settings)
    report = inspect_csv_file(dataset_path, settings)
    settings.raw_dataset_inspection_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    settings.raw_dataset_inspection_markdown_path.write_text(
        build_raw_dataset_inspection_markdown(report),
        encoding="utf-8",
    )
    get_logger().info(
        "Saved raw dataset inspection to %s and %s",
        settings.raw_dataset_inspection_json_path.relative_to(settings.project_root),
        settings.raw_dataset_inspection_markdown_path.relative_to(settings.project_root),
    )
    return report


def discover_datasets(settings: Settings) -> list[dict[str, object]]:
    """Inspect every CSV in data/raw/ and summarize dataset availability."""

    logger = get_logger()
    csv_paths = sorted(settings.raw_data_dir.glob("*.csv"))
    reports = [inspect_csv_file(path, settings, include_preview=False) for path in csv_paths]

    frame = pd.DataFrame(
        [
            {
                "dataset_path": report["dataset_path"],
                "file_size_mb": report["file_size_mb"],
                "raw_row_count": report["raw_row_count"],
                "raw_column_count": report["raw_column_count"],
                "raw_columns": "|".join(report["raw_columns"]),
                "has_author_steamid": report["author_steamid_exists_exactly"],
                "author_steamid_non_empty_count": report["author_steamid_non_empty_count"],
                "detected_mode": report["detected_mode"],
                "warnings": " ; ".join(report["warnings"]),
            }
            for report in reports
        ]
    )
    frame.to_csv(settings.dataset_discovery_report_csv_path, index=False)
    settings.dataset_discovery_report_json_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    settings.dataset_discovery_report_markdown_path.write_text(
        build_dataset_discovery_markdown(reports),
        encoding="utf-8",
    )
    logger.info(
        "Saved dataset discovery reports to %s, %s, and %s",
        settings.dataset_discovery_report_csv_path.relative_to(settings.project_root),
        settings.dataset_discovery_report_json_path.relative_to(settings.project_root),
        settings.dataset_discovery_report_markdown_path.relative_to(settings.project_root),
    )
    return reports


def inspect_csv_file(
    dataset_path: Path,
    settings: Settings,
    *,
    include_preview: bool = True,
) -> dict[str, object]:
    """Collect raw-file inspection details without full preprocessing."""

    raw_columns = read_dataset_header(dataset_path)
    resolution = build_column_resolution(raw_columns)
    count_raw_column = resolution["canonical_to_raw"].get(
        "review_id",
        resolution["raw_columns"][0] if resolution["raw_columns"] else "",
    )
    raw_row_count = count_csv_rows(dataset_path, count_raw_column) if count_raw_column else 0
    file_size_mb = round(dataset_path.stat().st_size / (1024 * 1024), 4)
    steamid_raw_column = resolution["canonical_to_raw"].get("author.steamid")
    app_id_raw_column = resolution["canonical_to_raw"].get("app_id")
    non_empty_steamid_count = 0
    null_steamid_count = 0
    unique_steamid_count = 0
    unique_game_count = 0
    masked_examples: list[str] = []
    preview_rows: list[dict[str, object]] = []

    if steamid_raw_column:
        steamid_summary = summarize_identifier_column(
            dataset_path,
            steamid_raw_column,
            track_unique=False,
        )
        non_empty_steamid_count = int(steamid_summary["non_empty_count"])
        null_steamid_count = int(steamid_summary["empty_count"])
        unique_steamid_count = int(steamid_summary["unique_count"])
        masked_examples = list(steamid_summary["masked_examples"])

    if app_id_raw_column:
        game_summary = summarize_identifier_column(dataset_path, app_id_raw_column)
        unique_game_count = int(game_summary["unique_count"])

    if include_preview:
        preview_rows = load_preview_rows(dataset_path, resolution)

    has_basic_columns = all(
        column in resolution["canonical_to_raw"] for column in BASIC_MODE_REQUIRED_COLUMNS
    )
    missing_expected_count = len(
        [
            column
            for column in settings.expected_full_schema_columns
            if column not in resolution["canonical_to_raw"]
        ]
    )
    likely_sample_or_incomplete = detect_likely_sample_or_incomplete(
        raw_row_count=raw_row_count,
        unique_games=unique_game_count,
        file_size_mb=file_size_mb,
        missing_expected_count=missing_expected_count,
        has_steamid_column=bool(steamid_raw_column),
        non_empty_steamid_count=non_empty_steamid_count,
    )
    warnings = list(resolution["column_name_warnings"])
    if likely_sample_or_incomplete:
        warnings.append(SAMPLE_WARNING_MESSAGE)

    detected_mode = "unknown_invalid"
    if steamid_raw_column and non_empty_steamid_count == 0:
        detected_mode = "invalid_or_empty_user_id"
    elif steamid_raw_column and non_empty_steamid_count > 0:
        detected_mode = "user_based_available"
    elif has_basic_columns:
        detected_mode = "scenario_only"
    if likely_sample_or_incomplete and detected_mode in {
        "scenario_only",
        "unknown_invalid",
    }:
        detected_mode = "sample_or_incomplete"

    return {
        "dataset_path": str(dataset_path),
        "file_size_mb": file_size_mb,
        "raw_row_count": raw_row_count,
        "raw_column_count": len(raw_columns),
        "raw_columns": raw_columns,
        "first_five_column_names_as_read_by_pandas": raw_columns[:5],
        "author_steamid_exists_exactly": "author.steamid" in raw_columns,
        "any_column_contains_steamid": bool(resolution["has_any_steamid_column"]),
        "any_column_contains_author": bool(resolution["has_any_author_column"]),
        "author_steamid_source_column": steamid_raw_column or "",
        "author_steamid_non_empty_count": non_empty_steamid_count,
        "author_steamid_null_count": null_steamid_count,
        "author_steamid_unique_non_empty_count": unique_steamid_count,
        "sample_author_steamid_values_masked": masked_examples,
        "first_three_raw_rows": preview_rows,
        "column_name_warnings": resolution["column_name_warnings"],
        "likely_sample_or_incomplete": likely_sample_or_incomplete,
        "warnings": warnings,
        "detected_mode": detected_mode,
    }


def read_dataset_header(dataset_path: Path) -> list[str]:
    """Read only the header row from a CSV file using pandas column parsing."""

    return [str(column) for column in pd.read_csv(dataset_path, nrows=0).columns.tolist()]


def count_csv_rows(dataset_path: Path, count_column_name: str) -> int:
    """Count data rows without loading the full dataset into memory."""

    count = 0
    for chunk in pd.read_csv(dataset_path, usecols=[count_column_name], chunksize=STREAM_CHUNK_SIZE):
        count += len(chunk)
    return int(count)


def summarize_identifier_column(
    dataset_path: Path,
    column_name: str,
    *,
    track_unique: bool = True,
) -> dict[str, object]:
    """Stream one identifier-like column and summarize it."""

    non_empty_count = 0
    empty_count = 0
    unique_values: set[str] | None = set() if track_unique else None
    masked_examples: list[str] = []

    for chunk in pd.read_csv(dataset_path, usecols=[column_name], chunksize=STREAM_CHUNK_SIZE):
        normalized = normalize_identifier_series(chunk[column_name])
        non_empty_mask = normalized.str.len() > 0
        non_empty_values = normalized[non_empty_mask]
        non_empty_count += int(non_empty_mask.sum())
        empty_count += int((~non_empty_mask).sum())
        if unique_values is not None:
            unique_values.update(non_empty_values.unique().tolist())
        for value in non_empty_values.tolist():
            masked = mask_identifier(value)
            if masked and masked not in masked_examples:
                masked_examples.append(masked)
            if len(masked_examples) >= 5:
                break

    return {
        "non_empty_count": non_empty_count,
        "empty_count": empty_count,
        "unique_count": len(unique_values) if unique_values is not None else 0,
        "masked_examples": masked_examples[:5],
    }


def normalize_identifier_series(values: pd.Series) -> pd.Series:
    """Normalize identifier-like values while preserving valid numeric ids."""

    normalized = values.map(normalize_identifier_value)
    return normalized.fillna("").astype(str)


def normalize_identifier_value(value: object) -> str:
    """Normalize a single identifier-like value."""

    if value is None or pd.isna(value):
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
        text = str(value).strip()
        return "" if text.lower() in {"nan", "none", "null", "<na>"} else text

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", maxsplit=1)[0]
    return text


def load_preview_rows(
    dataset_path: Path,
    resolution: dict[str, object],
) -> list[dict[str, object]]:
    """Load the first three raw rows for selected columns only."""

    canonical_to_raw = resolution["canonical_to_raw"]
    selected_raw_columns = [
        canonical_to_raw[column]
        for column in RAW_PREVIEW_COLUMNS
        if column in canonical_to_raw
    ]
    if not selected_raw_columns:
        return []

    preview_df = pd.read_csv(dataset_path, usecols=selected_raw_columns, nrows=3)
    raw_to_canonical = {
        raw_column: canonical
        for canonical, raw_column in canonical_to_raw.items()
        if raw_column in selected_raw_columns
    }
    preview_df = preview_df.rename(columns=raw_to_canonical)
    if "author.steamid" in preview_df.columns:
        preview_df["author.steamid"] = normalize_identifier_series(
            preview_df["author.steamid"]
        ).map(mask_identifier)
    return json.loads(preview_df.to_json(orient="records", force_ascii=False))


def detect_likely_sample_or_incomplete(
    *,
    raw_row_count: int,
    unique_games: int,
    file_size_mb: float,
    missing_expected_count: int,
    has_steamid_column: bool,
    non_empty_steamid_count: int,
) -> bool:
    """Flag files that still look like sample or incomplete inputs."""

    return bool(
        raw_row_count <= 100
        or unique_games <= 10
        or file_size_mb < 0.1
        or missing_expected_count >= 5
        or (has_steamid_column and non_empty_steamid_count == 0)
    )


def mask_identifier(value: str) -> str:
    """Mask identifier-like values for markdown-safe reporting."""

    text = normalize_identifier_value(value)
    if not text:
        return ""
    if len(text) <= 4:
        return "*" * len(text)
    return text[:4] + "*" * (len(text) - 4)


def build_schema_validation_markdown(report: dict[str, object]) -> str:
    """Render the schema validation report as concise markdown."""

    missing_lines = [f"- {column}" for column in report["missing_expected_columns"]] or ["- none"]
    extra_lines = [f"- {column}" for column in report["extra_columns"]] or ["- none"]
    warning_lines = [f"- {warning}" for warning in report["column_name_warnings"]] or ["- none"]
    return "\n".join(
        [
            "# Schema Validation Report",
            "",
            f"- Dataset path: `{report['dataset_path']}`",
            f"- Status: {report['status']}",
            f"- Required for basic mode present: {report['required_for_basic_mode_present']}",
            f"- Required for user mode present: {report['required_for_user_mode_present']}",
            f"- User id column detected: {report['user_id_column_detected']}",
            f"- Matched user id source column: `{report['matched_user_id_source_column']}`",
            "",
            "## Column Name Diagnostics",
            *warning_lines,
            "",
            "## Missing Expected Columns",
            *missing_lines,
            "",
            "## Extra Columns",
            *extra_lines,
            "",
        ]
    ) + "\n"


def build_raw_dataset_inspection_markdown(report: dict[str, object]) -> str:
    """Render the raw-dataset inspection report as markdown."""

    warning_lines = [f"- {warning}" for warning in report["warnings"]] or ["- none"]
    preview_frame = pd.DataFrame(report["first_three_raw_rows"])
    return "\n".join(
        [
            "# Raw Dataset Inspection",
            "",
            f"- Dataset path: `{report['dataset_path']}`",
            f"- File size (MB): {report['file_size_mb']}",
            f"- Raw row count: {report['raw_row_count']}",
            f"- Raw column count: {report['raw_column_count']}",
            f"- Exact `author.steamid` column present: {report['author_steamid_exists_exactly']}",
            f"- Any column contains `steamid`: {report['any_column_contains_steamid']}",
            f"- Any column contains `author`: {report['any_column_contains_author']}",
            f"- Non-empty `author.steamid` count: {report['author_steamid_non_empty_count']}",
            f"- Null/empty `author.steamid` count: {report['author_steamid_null_count']}",
            f"- Detected mode: {report['detected_mode']}",
            "",
            "## Warnings",
            *warning_lines,
            "",
            "## Raw Columns",
            *[f"- {column}" for column in report["raw_columns"]],
            "",
            "## Masked `author.steamid` Examples",
            *([f"- {value}" for value in report["sample_author_steamid_values_masked"]] or ["- none"]),
            "",
            "## First 3 Raw Rows",
            dataframe_to_markdown(preview_frame),
            "",
        ]
    ) + "\n"


def build_dataset_discovery_markdown(reports: list[dict[str, object]]) -> str:
    """Render the dataset discovery output as markdown."""

    if not reports:
        return "# Dataset Discovery\n\n_No CSV files found in `data/raw/`._\n"

    frame = pd.DataFrame(
        [
            {
                "dataset_path": report["dataset_path"],
                "file_size_mb": report["file_size_mb"],
                "raw_row_count": report["raw_row_count"],
                "raw_column_count": report["raw_column_count"],
                "has_author_steamid": report["author_steamid_exists_exactly"],
                "author_steamid_non_empty_count": report["author_steamid_non_empty_count"],
                "detected_mode": report["detected_mode"],
            }
            for report in reports
        ]
    )
    lines = [
        "# Dataset Discovery",
        "",
        dataframe_to_markdown(frame),
        "",
        "## Warnings",
    ]
    warnings = []
    for report in reports:
        if report["warnings"]:
            warnings.append(f"- `{report['dataset_path']}`: {' ; '.join(report['warnings'])}")
    lines.extend(warnings or ["- none"])
    lines.extend(["", "## Column Lists"])
    for report in reports:
        lines.append(f"### `{report['dataset_path']}`")
        lines.extend(f"- {column}" for column in report["raw_columns"])
        lines.append("")
    return "\n".join(lines) + "\n"


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small dataframe as markdown."""

    if frame.empty:
        return "_No rows available._"
    headers = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    return "\n".join(rows)


def load_external_scenarios(path: Path | None) -> list[dict[str, object]]:
    """Load external scenario definitions from JSONL or CSV if configured."""

    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found at '{path}'.")

    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        records = frame.to_dict(orient="records")
        return [normalize_external_scenario(record) for record in records]

    raise ValueError("Scenario file must use .jsonl or .csv format.")


def normalize_external_scenario(record: dict[str, object]) -> dict[str, object]:
    """Normalize list-like fields from CSV rows into Python lists."""

    list_fields = {
        "seed_game_ids",
        "excluded_game_ids",
        "ground_truth_game_ids",
        "candidate_game_ids",
    }

    normalized: dict[str, object] = {}
    for key, value in record.items():
        if key in list_fields:
            normalized[key] = _coerce_list(value)
        else:
            normalized[key] = value
    return normalized


def _coerce_list(value: object) -> list[str]:
    """Convert common serialized list formats to a list of strings."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            parsed = ast.literal_eval(stripped)
            return [str(item) for item in parsed]
        separator = "|" if "|" in stripped else ","
        return [item.strip() for item in stripped.split(separator) if item.strip()]
    return [str(value)]
