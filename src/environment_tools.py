
from __future__ import annotations

from importlib import import_module
from pathlib import Path
import json
import os
import platform
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")


REQUIRED_IMPORTS: dict[str, str] = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "openai": "openai",
    "pydantic": "pydantic",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
}


def run_environment_check(
    project_root: Path,
    dataset_path_override: Path | None = None,
) -> dict[str, object]:

    results_dir = project_root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report = build_environment_report(
        project_root,
        dataset_path_override=dataset_path_override,
    )

    json_path = results_dir / "environment_check.json"
    markdown_path = results_dir / "environment_check.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(
        build_environment_check_markdown(report),
        encoding="utf-8",
    )
    return report


def build_environment_report(
    project_root: Path,
    dataset_path_override: Path | None = None,
) -> dict[str, object]:

    checks: list[dict[str, str]] = []
    cwd = Path.cwd().resolve()

    python_ok = sys.version_info >= (3, 11)
    checks.append(
        check_record(
            name="python_version",
            status="ok" if python_ok else "error",
            message=(
                f"Python {platform.python_version()} detected."
                if python_ok
                else f"Python {platform.python_version()} detected. Python 3.11 or higher is required."
            ),
        )
    )

    package_statuses: dict[str, dict[str, str]] = {}
    for module_name, package_name in REQUIRED_IMPORTS.items():
        try:
            import_module(module_name)
            package_statuses[package_name] = {"status": "ok", "detail": "importable"}
            checks.append(
                check_record(
                    name=f"package:{package_name}",
                    status="ok",
                    message=f"{package_name} is importable.",
                )
            )
        except ImportError as exc:
            package_statuses[package_name] = {"status": "error", "detail": str(exc)}
            checks.append(
                check_record(
                    name=f"package:{package_name}",
                    status="error",
                    message=f"{package_name} could not be imported: {exc}",
                )
            )

    project_root_ok = all(
        (cwd / marker).exists() for marker in ("main.py", "src", "requirements.txt")
    )
    checks.append(
        check_record(
            name="working_directory",
            status="ok" if project_root_ok else "warning",
            message=(
                f"Current working directory looks like the project root: {cwd}"
                if project_root_ok
                else (
                    f"Current working directory is {cwd}. Run commands from the project root "
                    f"at {project_root} for predictable relative paths."
                )
            ),
        )
    )

    directories = [
        project_root / "data" / "raw",
        project_root / "data" / "processed",
        project_root / "data" / "results",
        project_root / "reports",
        project_root / "reports" / "figures",
        project_root / "data" / "scenarios",
    ]
    directory_statuses: dict[str, str] = {}
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            directory_statuses[str(directory.relative_to(project_root))] = "ok"
            checks.append(
                check_record(
                    name=f"directory:{directory.relative_to(project_root)}",
                    status="ok",
                    message=f"{directory.relative_to(project_root)} is available.",
                )
            )
        except OSError as exc:
            directory_statuses[str(directory.relative_to(project_root))] = f"error: {exc}"
            checks.append(
                check_record(
                    name=f"directory:{directory.relative_to(project_root)}",
                    status="error",
                    message=f"Could not create {directory.relative_to(project_root)}: {exc}",
                )
            )

    env_example_path = project_root / ".env.example"
    checks.append(
        check_record(
            name=".env.example",
            status="ok" if env_example_path.exists() else "error",
            message=(
                ".env.example is present."
                if env_example_path.exists()
                else ".env.example is missing."
            ),
        )
    )

    env_path = project_root / ".env"
    env_exists = env_path.exists()
    checks.append(
        check_record(
            name=".env",
            status="ok" if env_exists else "warning",
            message=(
                ".env is present."
                if env_exists
                else ".env is missing. Copy .env.example to .env before running a real experiment."
            ),
        )
    )

    env_values = load_simple_env(env_path if env_exists else env_example_path)
    dataset_setting = env_values.get("STEAM_REVIEWS_CSV", "data/raw/steam_reviews.csv").strip()
    dataset_path = dataset_path_override or (
        Path(dataset_setting)
        if Path(dataset_setting).is_absolute()
        else project_root / dataset_setting
    )
    dataset_exists = dataset_path.exists()
    checks.append(
        check_record(
            name="raw_dataset",
            status="ok" if dataset_exists else "warning",
            message=(
                f"Configured dataset path exists: {dataset_path}"
                if dataset_exists
                else (
                    f"Configured dataset path does not exist: {dataset_path}. "
                    "Place the Steam Reviews CSV there or update STEAM_REVIEWS_CSV in .env."
                )
            ),
        )
    )

    openai_api_key = env_values.get("OPENAI_API_KEY", "").strip()
    openai_model = env_values.get("OPENAI_MODEL", "").strip()
    llm_ready = bool(openai_api_key and openai_model)
    checks.append(
        check_record(
            name="openai_credentials",
            status="ok" if llm_ready else "warning",
            message=(
                "OPENAI_API_KEY and OPENAI_MODEL are configured."
                if llm_ready
                else "OpenAI credentials are incomplete. LLM reranking will be skipped."
            ),
        )
    )

    summary = {
        "ok": sum(1 for check in checks if check["status"] == "ok"),
        "warnings": sum(1 for check in checks if check["status"] == "warning"),
        "errors": sum(1 for check in checks if check["status"] == "error"),
    }

    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "project_root": str(project_root),
        "working_directory": str(cwd),
        "configured_dataset_path": str(dataset_path),
        "llm_reranking_ready": llm_ready,
        "checks": checks,
        "package_statuses": package_statuses,
        "directory_statuses": directory_statuses,
        "summary": summary,
    }


def check_record(name: str, status: str, message: str) -> dict[str, str]:

    return {"name": name, "status": status, "message": message}


def load_simple_env(path: Path) -> dict[str, str]:

    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def build_environment_check_markdown(report: dict[str, object]) -> str:

    lines = [
        "# Environment Check",
        "",
        "## Summary",
        f"- Python version: {report['python_version']}",
        f"- Python executable: `{report['python_executable']}`",
        f"- Project root: `{report['project_root']}`",
        f"- Working directory: `{report['working_directory']}`",
        f"- Configured dataset path: `{report['configured_dataset_path']}`",
        f"- LLM reranking ready: {report['llm_reranking_ready']}",
        f"- OK checks: {report['summary']['ok']}",
        f"- Warnings: {report['summary']['warnings']}",
        f"- Errors: {report['summary']['errors']}",
        "",
        "## Checks",
        "",
        "| name | status | message |",
        "| --- | --- | --- |",
    ]

    for check in report["checks"]:
        lines.append(
            f"| {check['name']} | {check['status']} | {check['message'].replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def find_missing_runtime_dependencies() -> list[str]:

    missing: list[str] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            import_module(module_name)
        except ImportError:
            missing.append(module_name)
    return missing
