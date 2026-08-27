"""Performance-blind KernelBench task selection and checksum-frozen manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from openkernelforge.tasks.kernelbench_l1 import (
    estimate_task_memory,
    generate_inputs_for_memory_estimate,
    load_kernelbench_l1_tasks,
)


@dataclass(frozen=True)
class SelectionPaths:
    manifest: Path
    csv: Path
    checksum: Path


def freeze_kernelbench_selection(
    protocol_path: str | Path,
    kernelbench_dir: str | Path,
    *,
    output_root: str | Path | None = None,
    replace: bool = False,
) -> SelectionPaths:
    protocol_file = Path(protocol_path).resolve()
    protocol = yaml.safe_load(protocol_file.read_text(encoding="utf-8")) or {}
    kb_config = protocol.get("kernelbench") or {}
    freeze_config = kb_config.get("freeze") or {}
    expected_commit = str(kb_config.get("commit") or "")
    checkout = Path(kernelbench_dir).expanduser().resolve()
    actual_commit = _git_commit(checkout)
    if expected_commit and actual_commit != expected_commit:
        raise RuntimeError(
            f"KernelBench commit mismatch: expected {expected_commit}, found {actual_commit}"
        )

    if output_root is None:
        manifest_path = Path(str(freeze_config["manifest"]))
        csv_path = Path(str(freeze_config["csv"]))
        checksum_path = Path(str(freeze_config["checksum"]))
    else:
        root = Path(output_root)
        manifest_path = root / "task_selection_manifest.json"
        csv_path = root / "task_selection_manifest.csv"
        checksum_path = root / "task_selection_manifest.sha256"
    paths = SelectionPaths(manifest_path, csv_path, checksum_path)
    existing = [path for path in (paths.manifest, paths.csv, paths.checksum) if path.exists()]
    if existing and not replace:
        raise FileExistsError(
            "Task selection is already frozen; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    target_tasks = int(kb_config.get("target_tasks", 50))
    family_order = [str(item) for item in kb_config.get("family_order", [])]
    preflight = kb_config.get("memory_preflight") or {}
    allow_cpu_fallback = bool(preflight.get("allow_cpu_materialization_fallback", False))
    max_numel = _optional_int(preflight.get("max_input_numel"))
    max_total_bytes = _optional_int(preflight.get("max_total_input_bytes"))
    max_peak_mb = _optional_float(preflight.get("max_estimated_known_peak_mb"))

    tasks = load_kernelbench_l1_tasks(checkout, stratify_by_family=False)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        row: dict[str, Any] = {
            "task_id": task.task_id,
            "task_name": task.name,
            "family": str(task.metadata.get("op_family", "kernelbench_l1")),
            "source_relative_path": task.metadata.get("source_relative_path"),
            "source_sha256": _sha256_file(Path(str(task.metadata["source_path"]))),
            "candidate_or_performance_fields_read": False,
            "feasible": False,
            "selected": False,
            "skip_reason": "",
        }
        try:
            inputs = generate_inputs_for_memory_estimate(
                task,
                seed=0,
                dtype=torch.float32,
                allow_cpu_fallback=allow_cpu_fallback,
            )
            estimate = estimate_task_memory(task, inputs)
            row.update(
                {
                    "input_shapes": [tensor["shape"] for tensor in estimate["tensors"]],
                    "input_dtypes": [tensor["dtype"] for tensor in estimate["tensors"]],
                    "total_input_bytes": int(estimate["total_bytes"]),
                    "max_tensor_numel": int(estimate["max_tensor_numel"]),
                    "estimated_known_peak_mb": float(estimate["estimated_known_peak_mb"]),
                    "estimate_scope": estimate["estimate_scope"],
                }
            )
            reason = _memory_skip_reason(
                estimate,
                max_numel=max_numel,
                max_total_bytes=max_total_bytes,
                max_peak_mb=max_peak_mb,
            )
            if reason:
                row["skip_reason"] = reason
            else:
                row["feasible"] = True
        except Exception as exc:
            row["skip_reason"] = f"MEMORY_PREFLIGHT_FAILED: {type(exc).__name__}: {exc}"
        rows.append(row)

    selected_ids = _family_round_robin_select(rows, target_tasks=target_tasks, family_order=family_order)
    if len(selected_ids) < target_tasks:
        raise RuntimeError(
            f"Only {len(selected_ids)} tasks passed performance-blind feasibility; "
            f"the frozen target is {target_tasks}. No manifest was written."
        )
    selected_rank = {task_id: index + 1 for index, task_id in enumerate(selected_ids)}
    for row in rows:
        rank = selected_rank.get(str(row["task_id"]))
        if rank is not None:
            row["selected"] = True
            row["selection_rank"] = rank
        else:
            row["selection_rank"] = ""
            if row["feasible"] and not row["skip_reason"]:
                row["skip_reason"] = "FEASIBLE_NOT_SELECTED_AFTER_TARGET_REACHED"

    protocol_sha = _sha256_file(protocol_file)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_CANDIDATE_PERFORMANCE",
        "study_id": (protocol.get("study") or {}).get("id"),
        "protocol_path": str(protocol_file),
        "protocol_sha256": protocol_sha,
        "kernelbench_repository": kb_config.get("repository"),
        "kernelbench_checkout": str(checkout),
        "kernelbench_commit": actual_commit,
        "selection_rule": kb_config.get("selection_rule"),
        "selection_blinding": kb_config.get("selection_blinding"),
        "family_order": family_order,
        "target_tasks": target_tasks,
        "loaded_tasks": len(rows),
        "feasible_tasks": sum(bool(row["feasible"]) for row in rows),
        "selected_task_ids": selected_ids,
        "selected_family_counts": _family_counts(rows, selected_only=True),
        "feasible_family_counts": _family_counts(rows, feasible_only=True),
        "skipped_reason_counts": _reason_counts(rows),
        "memory_preflight": preflight,
        "rows": rows,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    for path in (paths.manifest, paths.csv, paths.checksum):
        path.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_bytes(manifest_bytes)
    _write_csv(paths.csv, rows)
    paths.checksum.write_text(f"{manifest_sha}  {paths.manifest.name}\n", encoding="utf-8")
    return paths


def _family_round_robin_select(
    rows: Sequence[dict[str, Any]],
    *,
    target_tasks: int,
    family_order: Sequence[str],
) -> list[str]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("feasible"):
            buckets.setdefault(str(row["family"]), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: (str(row.get("source_relative_path")), str(row["task_id"])))
    order = list(dict.fromkeys([*family_order, *sorted(buckets)]))
    selected: list[str] = []
    while len(selected) < target_tasks and any(buckets.get(family) for family in order):
        for family in order:
            bucket = buckets.get(family) or []
            if bucket and len(selected) < target_tasks:
                selected.append(str(bucket.pop(0)["task_id"]))
    return selected


def _memory_skip_reason(
    estimate: dict[str, Any],
    *,
    max_numel: int | None,
    max_total_bytes: int | None,
    max_peak_mb: float | None,
) -> str | None:
    if max_numel is not None and int(estimate["max_tensor_numel"]) > max_numel:
        return "ESTIMATED_MEMORY_TOO_LARGE:max_input_numel"
    if max_total_bytes is not None and int(estimate["total_bytes"]) > max_total_bytes:
        return "ESTIMATED_MEMORY_TOO_LARGE:max_total_input_bytes"
    if max_peak_mb is not None and float(estimate["estimated_known_peak_mb"]) > max_peak_mb:
        return "ESTIMATED_MEMORY_TOO_LARGE:max_estimated_known_peak_mb"
    return None


def _git_commit(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "selection_rank",
        "task_id",
        "task_name",
        "family",
        "source_relative_path",
        "source_sha256",
        "feasible",
        "selected",
        "skip_reason",
        "input_shapes",
        "input_dtypes",
        "total_input_bytes",
        "max_tensor_numel",
        "estimated_known_peak_mb",
        "estimate_scope",
        "candidate_or_performance_fields_read",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _family_counts(
    rows: Sequence[dict[str, Any]],
    *,
    selected_only: bool = False,
    feasible_only: bool = False,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if selected_only and not row.get("selected"):
            continue
        if feasible_only and not row.get("feasible"):
            continue
        family = str(row["family"])
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def _reason_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("skip_reason") or "")
        if reason:
            key = reason.split(":", 1)[0]
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
