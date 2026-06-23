"""Shared helpers for small file and text operations."""

from __future__ import annotations

from pathlib import Path
import json
import logging
import re
from typing import Any, Iterable


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create directories if they do not already exist."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""

    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries to a JSONL file."""

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def truncate_text(text: str, max_chars: int) -> str:
    """Trim text while keeping words readable."""

    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def get_logger(name: str = "steam_recommender") -> logging.Logger:
    """Create a simple console logger."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_text(path: Path) -> str:
    """Read a plain-text file."""

    return path.read_text(encoding="utf-8")


def model_to_dict(model: Any) -> dict[str, Any]:
    """Convert a Pydantic model to a plain dictionary across versions."""

    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

