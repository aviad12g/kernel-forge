#!/usr/bin/env python3
"""Run the calibrated near-threshold multiplicity stress test on CUDA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from openkernelforge.reports.holdout_confirmation import TimingBlock
from openkernelforge.reports.selection_multiplicity import (
    analyze_selection_multiplicity,
    write_multiplicity_csv,
)
from openkernelforge.utils.env_probe import TRITON_EXECUTION_OK, probe_environment


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/workshop2026_near_threshold_multiplicity_protocol.yaml",
    )
    parser.add_argument(
        "--candidate-pool-manifest",
        default="artifacts/workshop2026/near_threshold_multiplicity/candidate_pool_manifest.json",
    )
    parser.add_argument(
        "--selected-manifest",
        default="artifacts/workshop2026/near_threshold_multiplicity/selected_candidate_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/workshop2026/near_threshold_multiplicity/campaign",
    )
    parser.add_argument("--stage", choices=("calibration", "screening", "confirmation", "all"), default="all")
    parser.add_argument("--max-gpu-hours", type=float, default=5.0)
    parser.add_argument("--wait-for-separation", action="store_true")
    args = parser.parse_args()
    _require_cuda_linux()

    protocol_path = _repo_path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    configured_cap = float(protocol["budget"]["maximum_recorded_gpu_worker_hours"])
    if args.max_gpu_hours <= 0 or args.max_gpu_hours > configured_cap:
        raise ValueError(f"max GPU hours must be in (0, {configured_cap}]")
    pool_path = _repo_path(args.candidate_pool_manifest)
    selected_path = _repo_path(args.selected_manifest)
    root = _repo_path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    pool = _load_and_validate_manifest(
        pool_path,
        protocol_path=protocol_path,
        allowed_statuses={"FROZEN_BEFORE_ANY_TIMING"},
        expected_per_task=int(protocol["candidates"]["variants_per_task"]),
    )

    if args.stage in {"calibration", "all"}:
        _run_calibration(
            root,
            protocol=protocol,
            protocol_path=protocol_path,
            manifest=pool,
            manifest_path=pool_path,
            max_gpu_hours=args.max_gpu_hours,
        )
        _freeze_selected_manifest(
            root,
            protocol=protocol,
            protocol_path=protocol_path,
            pool=pool,
            pool_path=pool_path,
            selected_path=selected_path,
        )
        _write_state(root, "calibration_complete")
        _write_sha256_manifest(root)
        if args.stage == "calibration":
            print(f"near-threshold calibration complete: {root}")
            return 0

    selected = _load_and_validate_manifest(
        selected_path,
        protocol_path=protocol_path,
        allowed_statuses={"FROZEN_AFTER_DISJOINT_CALIBRATION_BEFORE_PRIMARY_TIMING"},
        expected_per_task=int(protocol["candidates"]["selected_candidates_per_task"]),
    )
    if args.stage in {"screening", "all"}:
        _run_screening(
            root,
            protocol=protocol,
            protocol_path=protocol_path,
            manifest=selected,
            manifest_path=selected_path,
            max_gpu_hours=args.max_gpu_hours,
        )
        _write_screening_lock(root, protocol)
        _write_state(root, "screening_complete")
        _write_sha256_manifest(root)
        if args.stage == "screening":
            print(f"near-threshold screening complete: {root}")
            return 0

    if args.stage in {"confirmation", "all"}:
        _enforce_screening_separation(
            root,
            wait=args.wait_for_separation,
        )
        _run_confirmation(
            root,
            protocol=protocol,
            protocol_path=protocol_path,
            manifest=selected,
            manifest_path=selected_path,
            max_gpu_hours=args.max_gpu_hours,
        )
        _summarize_primary(root, protocol=protocol, selected=selected)
        _write_state(root, "complete")
        _write_sha256_manifest(root)
    print(f"near-threshold multiplicity campaign complete: {root}")
    return 0


def select_calibrated_candidates(
    candidate_speedups: dict[str, float],
    *,
    count: int,
    lower: float,
    upper: float,
) -> tuple[list[str], dict[str, str]]:
    """Apply the prespecified window-first deterministic selection rule."""

    if count <= 0 or not (0 < lower <= 1.0 <= upper):
        raise ValueError("invalid near-threshold selection configuration")
    if len(candidate_speedups) < count:
        raise ValueError("not enough calibrated candidates to freeze the primary subset")
    ranked = sorted(
        candidate_speedups,
        key=lambda candidate_id: (
            abs(math.log(candidate_speedups[candidate_id])),
            candidate_id,
        ),
    )
    in_window = [
        candidate_id
        for candidate_id in ranked
        if lower <= candidate_speedups[candidate_id] <= upper
    ]
    selected = in_window[:count]
    for candidate_id in ranked:
        if len(selected) == count:
            break
        if candidate_id not in selected:
            selected.append(candidate_id)
    reasons = {
        candidate_id: (
            "within_preregistered_window"
            if lower <= candidate_speedups[candidate_id] <= upper
            else "closest_absolute_log_distance_fallback"
        )
        for candidate_id in selected
    }
    return selected, reasons


def _run_calibration(
    root: Path,
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    max_gpu_hours: float,
) -> None:
    processes = int(protocol["calibration"]["fresh_processes"])
    seed_base = int(protocol["calibration"]["process_seed_base"])
    for task_index, task_id in enumerate(_task_ids(protocol)):
        for process_index in range(processes):
            _check_budget(root, max_gpu_hours)
            _run_job(
                root=root / "calibration",
                protocol_path=protocol_path,
                manifest_path=manifest_path,
                phase="calibration",
                task_id=task_id,
                process_id=f"c{process_index:02d}",
                seed=seed_base + task_index * 10_007 + process_index * 101,
                candidates=manifest["tasks"][task_id],
            )


def _freeze_selected_manifest(
    root: Path,
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    pool: dict[str, Any],
    pool_path: Path,
    selected_path: Path,
) -> None:
    expected = int(protocol["candidates"]["selected_candidates_per_task"])
    lower, upper = [float(value) for value in protocol["calibration"]["selection_window_speedup"]]
    calibration_rows: list[dict[str, Any]] = []
    selected_tasks: dict[str, list[dict[str, Any]]] = {}
    selected_reasons: dict[str, dict[str, str]] = {}
    for task_id in _task_ids(protocol):
        speedups = _calibration_speedups(
            root,
            task_id=task_id,
            processes=int(protocol["calibration"]["fresh_processes"]),
        )
        selected_ids, reasons = select_calibrated_candidates(
            speedups,
            count=expected,
            lower=lower,
            upper=upper,
        )
        minimum_in_window = int(
            protocol["calibration"].get("minimum_candidates_in_window_per_task_to_advance", 0)
        )
        in_window_count = sum(lower <= speedup <= upper for speedup in speedups.values())
        if in_window_count < minimum_in_window:
            raise RuntimeError(
                f"{task_id} has {in_window_count} calibrated candidates inside "
                f"[{lower}, {upper}]; {minimum_in_window} are required to advance"
            )
        selected_reasons[task_id] = reasons
        records = {str(item["candidate_id"]): item for item in pool["tasks"][task_id]}
        selected_tasks[task_id] = [records[candidate_id] for candidate_id in selected_ids]
        for candidate_id, speedup in sorted(speedups.items()):
            calibration_rows.append(
                {
                    "task_id": task_id,
                    "candidate_id": candidate_id,
                    "calibration_speedup": speedup,
                    "inside_window": lower <= speedup <= upper,
                    "selected_for_primary": candidate_id in reasons,
                    "selection_reason": reasons.get(candidate_id, "not_selected"),
                }
            )
    selection_csv = root / "calibration_selection.csv"
    _write_dict_csv(selection_csv, calibration_rows)
    calibration_sha = _sha256_file(selection_csv)
    if selected_path.exists():
        existing = json.loads(selected_path.read_text(encoding="utf-8"))
        if (
            existing.get("status")
            != "FROZEN_AFTER_DISJOINT_CALIBRATION_BEFORE_PRIMARY_TIMING"
            or existing.get("protocol_sha256") != _sha256_file(protocol_path)
            or existing.get("parent_candidate_pool_sha256") != _sha256_file(pool_path)
            or existing.get("calibration_selection_sha256") != calibration_sha
            or existing.get("tasks") != selected_tasks
        ):
            raise RuntimeError("frozen selected manifest disagrees with calibration artifacts")
        return
    payload = {
        "schema_version": 1,
        "status": "FROZEN_AFTER_DISJOINT_CALIBRATION_BEFORE_PRIMARY_TIMING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": protocol["study"]["id"],
        "protocol_path": _repo_relative(protocol_path),
        "protocol_sha256": _sha256_file(protocol_path),
        "candidate_source": pool["candidate_source"],
        "parent_candidate_pool_manifest": _repo_relative(pool_path),
        "parent_candidate_pool_sha256": _sha256_file(pool_path),
        "calibration_selection_path": _repo_relative(selection_csv),
        "calibration_selection_sha256": calibration_sha,
        "calibration_excluded_from_primary_analysis": True,
        "selection_window_speedup": [lower, upper],
        "selection_rule": protocol["calibration"]["selection_rule"],
        "selection_reasons": selected_reasons,
        "tasks": selected_tasks,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(text, encoding="utf-8")
    selected_path.with_suffix(".sha256").write_text(
        f"{_sha256_file(selected_path)}  {selected_path.name}\n",
        encoding="utf-8",
    )


def _run_screening(
    root: Path,
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    max_gpu_hours: float,
) -> None:
    for task_index, task_id in enumerate(_task_ids(protocol)):
        _check_budget(root, max_gpu_hours)
        _run_job(
            root=root / "screening",
            protocol_path=protocol_path,
            manifest_path=manifest_path,
            phase="screening",
            task_id=task_id,
            process_id="screening",
            seed=int(protocol["screening"]["process_seed"]) + task_index * 101,
            candidates=manifest["tasks"][task_id],
        )


def _run_confirmation(
    root: Path,
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    max_gpu_hours: float,
) -> None:
    process_count = int(protocol["confirmation"]["fresh_processes"])
    seed_base = int(protocol["confirmation"]["process_seed_base"])
    for task_index, task_id in enumerate(_task_ids(protocol)):
        for process_index in range(process_count):
            _check_budget(root, max_gpu_hours)
            _run_job(
                root=root / "confirmation",
                protocol_path=protocol_path,
                manifest_path=manifest_path,
                phase="confirmation",
                task_id=task_id,
                process_id=f"p{process_index:02d}",
                seed=seed_base + task_index * 10_007 + process_index * 101,
                candidates=manifest["tasks"][task_id],
            )


def _summarize_primary(
    root: Path,
    *,
    protocol: dict[str, Any],
    selected: dict[str, Any],
) -> None:
    process_count = int(protocol["confirmation"]["fresh_processes"])
    records = _collect_primary_records(
        root,
        _task_ids(protocol),
        process_count=process_count,
    )
    blocks_path = root / "primary_timing_blocks.csv"
    with blocks_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TimingBlock.__annotations__))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    analysis = protocol["analysis"]
    rows = analyze_selection_multiplicity(
        records,
        budgets=[int(value) for value in analysis["candidate_budgets"]],
        resamples=int(analysis["resamples"]),
        seed=int(analysis["seed"]),
        practical_margin=float(analysis["practical_speedup_margin"]),
        bootstrap_samples=int(analysis["bootstrap_samples"]),
        required_confirmation_processes=process_count,
    )
    write_multiplicity_csv(root / "selection_multiplicity.csv", rows)
    lines = [
        "# Near-Threshold Candidate-Budget Multiplicity",
        "",
        "Calibration selected candidates before primary screening. Calibration timings are excluded from every primary estimate.",
        "",
        f"- Frozen primary candidates: {sum(len(value) for value in selected['tasks'].values())}",
        f"- Recorded worker hours: {_recorded_worker_hours(root):.3f}",
        "",
        "| K | Tasks | Apparent wins | Confirmed wins | Median log optimism | 95% task-bootstrap interval |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.candidate_budget} | {row.eligible_tasks} | "
            f"{row.apparent_win_rate:.4f} | {row.confirmed_win_rate:.4f} | "
            f"{row.median_selection_optimism_log:.6f} | "
            f"[{row.selection_optimism_log_ci_lower:.6f}, {row.selection_optimism_log_ci_upper:.6f}] |"
        )
    (root / "near_threshold_multiplicity.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _calibration_speedups(
    root: Path,
    *,
    task_id: str,
    processes: int,
) -> dict[str, float]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for index in range(processes):
        process_id = f"c{index:02d}"
        path = root / "calibration" / task_id / process_id / "result.json"
        if not path.exists():
            raise RuntimeError(f"missing calibration result: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "completed":
            raise RuntimeError(f"incomplete calibration result: {path}")
        for candidate in data.get("candidate_results", []):
            if candidate.get("status") != "completed":
                continue
            values = []
            for block in (candidate.get("paired_timing") or {}).get("blocks", []):
                timings = block["median_ms_per_launch"]
                values.append(math.log(float(timings["eager"]) / float(timings["candidate"])))
            if values:
                grouped.setdefault(str(candidate["candidate_id"]), {})[process_id] = values
    result: dict[str, float] = {}
    for candidate_id, by_process in grouped.items():
        if len(by_process) != processes:
            continue
        process_medians = [statistics.median(by_process[key]) for key in sorted(by_process)]
        result[candidate_id] = math.exp(statistics.median(process_medians))
    return result


def _collect_primary_records(
    root: Path,
    task_ids: list[str],
    *,
    process_count: int,
) -> list[TimingBlock]:
    records: list[TimingBlock] = []
    phases = [("screening", "screening")]
    phases.extend(("confirmation", f"p{index:02d}") for index in range(process_count))
    for task_id in task_ids:
        for phase, process_id in phases:
            path = root / phase / task_id / process_id / "result.json"
            if not path.exists():
                raise RuntimeError(f"missing primary result: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))
            for candidate in data.get("candidate_results", []):
                valid = candidate.get("status") == "completed"
                for block in (candidate.get("paired_timing") or {}).get("blocks", []):
                    timings = block["median_ms_per_launch"]
                    records.append(
                        TimingBlock(
                            phase=phase,
                            task_id=task_id,
                            candidate_id=str(candidate["candidate_id"]),
                            process_id=process_id,
                            block_id=str(block["block_id"]),
                            eager_ms=float(timings["eager"]),
                            candidate_ms=float(timings["candidate"]),
                            correctness_passed=valid,
                            contract_passed=valid,
                        )
                    )
    return records


def _run_job(
    *,
    root: Path,
    protocol_path: Path,
    manifest_path: Path,
    phase: str,
    task_id: str,
    process_id: str,
    seed: int,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    job_dir = root / task_id / process_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / "job.json"
    result_path = job_dir / "result.json"
    job = {
        "protocol_path": str(protocol_path),
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": _sha256_file(manifest_path),
        "phase": phase,
        "task_id": task_id,
        "process_id": process_id,
        "seed": seed,
        "candidates": candidates,
    }
    text = json.dumps(job, indent=2) + "\n"
    if job_path.exists() and job_path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"existing job changed: {job_path}")
    job_path.write_text(text, encoding="utf-8")
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "completed":
            return result
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_multiplicity_worker.py",
            "--job",
            str(job_path),
            "--output",
            str(result_path),
        ],
        cwd=ROOT,
        text=True,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if completed.returncode or result.get("status") != "completed":
        raise RuntimeError(f"multiplicity worker failed: {result_path}")
    return result


def _load_and_validate_manifest(
    path: Path,
    *,
    protocol_path: Path,
    allowed_statuses: set[str],
    expected_per_task: int,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") not in allowed_statuses:
        raise RuntimeError(f"manifest is not in an allowed frozen state: {path}")
    if manifest.get("protocol_sha256") != _sha256_file(protocol_path):
        raise RuntimeError(f"protocol checksum mismatch: {path}")
    for task_id, candidates in manifest.get("tasks", {}).items():
        if len(candidates) != expected_per_task:
            raise RuntimeError(f"{task_id} has {len(candidates)} candidates; expected {expected_per_task}")
        for candidate in candidates:
            source = _repo_path(candidate["path"])
            metadata = _repo_path(candidate["metadata_path"])
            if _sha256_file(source) != candidate["sha256"]:
                raise RuntimeError(f"candidate source checksum mismatch: {task_id}")
            if _sha256_file(metadata) != candidate["metadata_sha256"]:
                raise RuntimeError(f"candidate metadata checksum mismatch: {task_id}")
    return manifest


def _write_screening_lock(root: Path, protocol: dict[str, Any]) -> None:
    path = root / "screening_confirmation_lock.json"
    if path.exists():
        return
    completed = datetime.now(timezone.utc)
    delay = int(protocol["confirmation"]["minimum_screen_to_confirmation_minutes"])
    payload = {
        "screening_completed_at_utc": completed.isoformat(),
        "confirmation_not_before_utc": (completed + timedelta(minutes=delay)).isoformat(),
        "minimum_separation_minutes": delay,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _enforce_screening_separation(root: Path, *, wait: bool) -> None:
    path = root / "screening_confirmation_lock.json"
    if not path.exists():
        raise RuntimeError("screening lock is missing; confirmation cannot start")
    lock = json.loads(path.read_text(encoding="utf-8"))
    not_before = datetime.fromisoformat(str(lock["confirmation_not_before_utc"]))
    remaining = (not_before - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        return
    if not wait:
        raise RuntimeError(
            f"confirmation is locked for another {remaining / 60.0:.1f} minutes; rerun with --wait-for-separation"
        )
    time.sleep(remaining)


def _recorded_worker_hours(root: Path) -> float:
    total = 0.0
    for path in root.glob("**/result.json"):
        try:
            total += max(0.0, float(json.loads(path.read_text(encoding="utf-8"))["elapsed_s"]))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return total / 3600.0


def _check_budget(root: Path, max_hours: float) -> None:
    used = _recorded_worker_hours(root)
    if used >= max_hours:
        raise RuntimeError(f"GPU time cap reached after {used:.3f} recorded worker hours")


def _write_state(root: Path, state: str) -> None:
    payload = {
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "recorded_worker_hours": _recorded_worker_hours(root),
    }
    (root / "campaign_state.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_sha256_manifest(root: Path) -> None:
    output = root / "SHA256SUMS"
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != output
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _task_ids(protocol: dict[str, Any]) -> list[str]:
    return [str(item) for item in protocol["tasks"]["ids"]]


def _require_cuda_linux() -> None:
    environment = probe_environment()
    if platform.system() == "Darwin" or environment.viability != TRITON_EXECUTION_OK:
        raise RuntimeError("near-threshold campaign requires a Linux CUDA/Triton environment")


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
