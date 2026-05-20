"""Small logging helpers for JSONL run artifacts."""

from __future__ import annotations

import json
import logging as py_logging
from pathlib import Path
from typing import Any


def get_logger(name: str = "openkernelforge") -> py_logging.Logger:
    logger = py_logging.getLogger(name)
    if not logger.handlers:
        handler = py_logging.StreamHandler()
        handler.setFormatter(py_logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(py_logging.INFO)
    return logger


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
