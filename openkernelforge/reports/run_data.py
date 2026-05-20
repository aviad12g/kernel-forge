"""Helpers for loading and enriching run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from openkernelforge.reports.summarize import load_results


def load_run_bundle(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    records = load_results(run_path)
    task_records = [
        record for record in records if record.get("record_type", "task_summary") != "candidate"
    ]
    candidate_records = [record for record in records if record.get("record_type") == "candidate"]
    environment = _load_json(run_path / "environment_probe.json")
    if not candidate_records:
        candidate_records = [
            candidate
            for record in task_records
            for candidate in record.get("candidate_records", [])
        ]
    attempts_by_key = {}
    tasks_by_id = {}
    for task_record in task_records:
        tasks_by_id[task_record.get("task_id")] = task_record
        for attempt in task_record.get("attempts", []):
            key = (task_record.get("task_id"), attempt.get("candidate_id"))
            attempts_by_key[key] = attempt

    enriched: list[dict[str, Any]] = []
    for candidate in candidate_records:
        merged = dict(candidate)
        attempt = attempts_by_key.get((candidate.get("task_id"), candidate.get("candidate_id")), {})
        if attempt:
            merged["attempt"] = attempt
            merged.setdefault("policy_result", attempt.get("policy"))
            merged.setdefault("verification_result", attempt.get("verification"))
            merged.setdefault("benchmark_result", attempt.get("benchmarks"))
            merged.setdefault("extraction", attempt.get("extraction"))
            merged.setdefault("error_log_path", attempt.get("error_log_path"))
        if environment:
            merged.setdefault("environment_probe", environment)
        enriched.append(merged)

    return {
        "run_dir": run_path,
        "records": records,
        "task_records": task_records,
        "candidate_records": enriched,
        "metadata": _load_json(run_path / "run_metadata.json"),
        "environment": environment,
        "config": _load_yaml(run_path / "config.yaml"),
        "summary_text": _read_optional(run_path / "summary.md"),
    }


def read_artifact(path_value: Any, *, run_dir: str | Path | None = None) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    if not path.exists() and run_dir is not None and not path.is_absolute():
        run_path = Path(run_dir)
        candidate = run_path / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _read_optional(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")
