
from __future__ import annotations

from pathlib import Path
import json
import logging
import re
from typing import Any, Iterable


def ensure_directories(paths: Iterable[Path]) -> None:

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:

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

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def truncate_text(text: str, max_chars: int) -> str:

    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def get_logger(name: str = "steam_recommender") -> logging.Logger:

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_text(path: Path) -> str:

    return path.read_text(encoding="utf-8")


def model_to_dict(model: Any) -> dict[str, Any]:

    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

