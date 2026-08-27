#!/usr/bin/env python3
"""Run screening and fresh-process confirmation from frozen artifacts only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from openkernelforge.reports.holdout_confirmation import (
    PromotionResult,
    TimingBlock,
    analyze_holdout_confirmation,
    select_screening_winners,
    summarize_campaign_aggregates,
    write_aggregate_artifacts,
    write_promotion_artifacts,
)
from openkernelforge.harness.paired_timing import cuda_environment_snapshot
from openkernelforge.utils.env_probe import TRITON_EXECUTION_OK, probe_environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/workshop2026_holdout_protocol.yaml")
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--kernelbench-dir", required=True)
    parser.add_argument("--campaign-validity", required=True)
    parser.add_argument("--output-dir", default="artifacts/workshop2026/holdout_campaign")
    parser.add_argument("--max-gpu-hours", type=float, default=5.0)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument(
        "--confirmation-wave",
        choices=("all", "wave1", "wave2"),
        default="all",
        help=(
            "Run all confirmation processes or one prespecified temporal wave. "
            "wave1 and wave2 preserve the same process IDs and seeds as all."
        ),
    )
    args = parser.parse_args()

    _require_cuda_runpod()
    started = time.monotonic()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(args.protocol).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    _validate_campaign_gate(Path(args.campaign_validity).resolve(), protocol_path, protocol)
    temporal_rule = protocol["confirmation"]["temporal_wave_rule"]
    if (
        temporal_rule["require_separate_invocations"]
        and args.confirmation_wave == "all"
        and not args.screen_only
    ):
        raise RuntimeError("confirmation must run as separate wave1 and wave2 invocations")
    task_manifest_path = Path(args.task_manifest).resolve()
    task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    kernelbench_dir = Path(args.kernelbench_dir).resolve()
    _validate_task_manifest(task_manifest, protocol_path, protocol, kernelbench_dir)
    candidate_manifest_path = Path(args.candidate_manifest).resolve()
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    task_manifest_sha = _sha256_file(task_manifest_path)
    _validate_candidate_manifest(candidate_manifest, task_manifest_sha)

    task_ids = list(task_manifest["selected_task_ids"])
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]
    raw_records: list[TimingBlock] = []
    screening_root = root / "screening"
    for task_index, task_id in enumerate(task_ids):
        _check_budget(root, args.max_gpu_hours)
        candidates = candidate_manifest["tasks"].get(task_id, [])
        expected = int(protocol["candidate_generation"]["candidates_per_task"])
        if len(candidates) != expected:
            raise RuntimeError(f"{task_id} has {len(candidates)} candidates; expected {expected}")
        output = _run_job(
            root=screening_root,
            protocol_path=protocol_path,
            task_manifest_path=task_manifest_path,
            task_manifest_sha=task_manifest_sha,
            kernelbench_dir=kernelbench_dir,
            phase="screening",
            task_id=task_id,
            process_id="screening",
            seed=int(protocol["screening"]["process_seed"]) + task_index * 101,
            candidates=candidates,
            allow_failed_result=True,
        )
        raw_records.extend(_timing_blocks_from_worker(output, phase="screening", task_id=task_id))

    _write_block_csv(root / "screening_blocks.csv", raw_records)
    winners = select_screening_winners(raw_records)
    winner_task_ids = {winner.task_id for winner in winners}
    invalid_screening_tasks = _screening_invalid_tasks(
        screening_root,
        task_ids,
        winner_task_ids=winner_task_ids,
    )
    winner_manifest = {
        "schema_version": 1,
        "status": "FROZEN_AFTER_SCREENING_BEFORE_CONFIRMATION",
        "task_manifest_sha256": task_manifest_sha,
        "candidate_manifest_sha256": _sha256_file(candidate_manifest_path),
        "winners": [asdict(winner) for winner in winners],
        "invalid_screening_tasks": invalid_screening_tasks,
    }
    winner_path = root / "screening_winners_frozen.json"
    if winner_path.exists():
        prior = json.loads(winner_path.read_text(encoding="utf-8"))
        if prior != winner_manifest:
            raise RuntimeError("frozen screening winner manifest changed; refusing confirmation")
    else:
        winner_path.write_text(json.dumps(winner_manifest, indent=2) + "\n", encoding="utf-8")
    if args.screen_only:
        _write_sha256_manifest(root)
        print(f"screening winners frozen: {winner_path}")
        return 0

    confirmation_root = root / "confirmation"
    process_count = int(protocol["confirmation"]["fresh_processes"])
    time_periods = int(protocol["confirmation"].get("spread_across_time_periods", 1))
    if time_periods != 2:
        raise RuntimeError(
            "the workshop runner currently requires exactly two confirmation time periods"
        )
    selected_process_indices = _confirmation_process_indices(
        process_count,
        wave=args.confirmation_wave,
        time_periods=time_periods,
    )
    if args.confirmation_wave == "wave2":
        _validate_wave2_start(root, temporal_rule)
    seed_base = int(protocol["confirmation"]["process_seed_base"])
    for task_index, winner in enumerate(winners):
        candidates = candidate_manifest["tasks"][winner.task_id]
        selected = [item for item in candidates if item["candidate_id"] == winner.candidate_id]
        if len(selected) != 1:
            raise RuntimeError(f"frozen candidate missing or duplicated for {winner.task_id}")
        for process_index in selected_process_indices:
            _check_budget(root, args.max_gpu_hours)
            output = _run_job(
                root=confirmation_root,
                protocol_path=protocol_path,
                task_manifest_path=task_manifest_path,
                task_manifest_sha=task_manifest_sha,
                kernelbench_dir=kernelbench_dir,
                phase="confirmation",
                task_id=winner.task_id,
                process_id=f"p{process_index:02d}",
                seed=seed_base + task_index * 10_007 + process_index * 101,
                candidates=selected,
                time_period=_confirmation_wave_for_index(
                    process_index,
                    process_count=process_count,
                    time_periods=time_periods,
                ),
            )
    if args.confirmation_wave == "wave1":
        _write_wave1_integrity_lock(
            root,
            confirmation_root,
            winners,
            selected_process_indices=selected_process_indices,
            minimum_separation_minutes=int(temporal_rule["minimum_separation_minutes"]),
        )
    confirmation_records, missing_jobs = _load_confirmation_records(
        confirmation_root,
        winners,
        process_count=process_count,
    )
    progress = {
        "schema_version": 1,
        "status": "complete" if not missing_jobs else "awaiting_confirmation_processes",
        "requested_wave": args.confirmation_wave,
        "fresh_processes_per_task": process_count,
        "completed_process_jobs": len(winners) * process_count - len(missing_jobs),
        "expected_process_jobs": len(winners) * process_count,
        "missing_jobs": missing_jobs,
    }
    (root / "confirmation_progress.json").write_text(
        json.dumps(progress, indent=2) + "\n",
        encoding="utf-8",
    )
    if missing_jobs:
        _write_sha256_manifest(root)
        print(
            f"confirmation {args.confirmation_wave} complete; "
            f"{len(missing_jobs)} process jobs remain before analysis"
        )
        return 0

    confirmation_records = [*raw_records, *confirmation_records]
    _write_block_csv(root / "all_timing_blocks.csv", confirmation_records)
    promotion = protocol["promotion"]
    results = analyze_holdout_confirmation(
        confirmation_records,
        winners,
        practical_margin=float(promotion["practical_speedup_margin"]),
        confidence_level=float(promotion["confidence_level"]),
        bootstrap_samples=int(promotion["bootstrap_samples"]),
        bootstrap_seed=int(promotion["bootstrap_seed"]),
        false_discovery_rate=float(promotion["false_discovery_rate"]),
        required_processes=int(promotion["required_confirmation_processes"]),
    )
    results.extend(
        _invalid_promotion_results(
            invalid_screening_tasks,
            practical_margin=float(promotion["practical_speedup_margin"]),
        )
    )
    results.sort(key=lambda item: task_ids.index(item.task_id))
    paths = write_promotion_artifacts(root / "analysis", winners, results)
    aggregate_config = protocol["aggregate_analysis"]
    aggregate = summarize_campaign_aggregates(
        results,
        practical_margin=float(promotion["practical_speedup_margin"]),
        confidence_level=float(promotion["confidence_level"]),
        bootstrap_samples=int(aggregate_config["task_bootstrap_samples"]),
        bootstrap_seed=int(aggregate_config["task_bootstrap_seed"]),
    )
    aggregate_paths = write_aggregate_artifacts(root / "analysis", aggregate)
    elapsed_hours = (time.monotonic() - started) / 3600.0
    recorded_worker_hours = _recorded_worker_hours(root)
    (root / "campaign_completion.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "elapsed_hours": elapsed_hours,
                "recorded_worker_hours": recorded_worker_hours,
                "task_count": len(task_ids),
                "fresh_processes_per_task": process_count,
                "analysis_paths": {key: str(path) for key, path in paths.items()},
                "aggregate_analysis_paths": {
                    key: str(path) for key, path in aggregate_paths.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_sha256_manifest(root)
    print(f"campaign complete in {elapsed_hours:.3f} hours: {root}")
    return 0


def _require_cuda_runpod() -> None:
    environment = probe_environment()
    if platform.system() == "Darwin" or environment.viability != TRITON_EXECUTION_OK:
        raise RuntimeError("holdout campaign requires a Linux CUDA/Triton environment")


def _validate_candidate_manifest(manifest: dict[str, Any], task_manifest_sha: str) -> None:
    if manifest.get("schema_version") != 2:
        raise RuntimeError("candidate manifest schema_version must be 2")
    if manifest.get("status") != "FROZEN_BEFORE_SCREENING":
        raise RuntimeError("candidate manifest is not frozen before screening")
    if manifest.get("task_selection_manifest_sha256") != task_manifest_sha:
        raise RuntimeError("candidate manifest does not match the frozen task selection")
    if manifest.get("provider_response_model_fields_preserved") is not True:
        raise RuntimeError("provider-returned model metadata must be preserved")
    if not isinstance(manifest.get("tasks"), dict):
        raise RuntimeError("candidate manifest must contain a tasks mapping")
    for task_id, candidates in manifest["tasks"].items():
        for candidate in candidates:
            required = {
                "path": "sha256",
                "prompt_path": "prompt_sha256",
                "raw_response_path": "raw_response_sha256",
                "metadata_path": "metadata_sha256",
            }
            for path_field, sha_field in required.items():
                path = Path(candidate[path_field]).resolve()
                if not path.exists() or _sha256_file(path) != candidate.get(sha_field):
                    raise RuntimeError(
                        f"candidate provenance mismatch: "
                        f"{task_id}/{candidate.get('candidate_id')}/{path_field}"
                    )
            if candidate.get("provider_response_model") in {None, "", "not_returned"}:
                raise RuntimeError(
                    f"provider response model missing: {task_id}/{candidate.get('candidate_id')}"
                )


def _validate_task_manifest(
    manifest: dict[str, Any],
    protocol_path: Path,
    protocol: dict[str, Any],
    kernelbench_dir: Path,
) -> None:
    if manifest.get("status") != "FROZEN_BEFORE_CANDIDATE_PERFORMANCE":
        raise RuntimeError("task selection manifest is not frozen")
    if manifest.get("protocol_sha256") != _sha256_file(protocol_path):
        raise RuntimeError("task manifest protocol hash is stale")
    expected_commit = str(protocol["kernelbench"]["commit"])
    actual_commit = subprocess.run(
        ["git", "-C", str(kernelbench_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if manifest.get("kernelbench_commit") != expected_commit or actual_commit != expected_commit:
        raise RuntimeError(
            f"KernelBench commit mismatch: expected {expected_commit}, found {actual_commit}"
        )


def _validate_campaign_gate(
    path: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
) -> None:
    if not path.exists():
        raise RuntimeError("campaign validity gate is missing; screening is forbidden")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or not all((gate.get("checks") or {}).values()):
        raise RuntimeError("campaign validity gate did not pass; screening is forbidden")
    if gate.get("study_id") != protocol.get("study", {}).get("id"):
        raise RuntimeError("campaign validity gate belongs to a different study")
    provenance = gate.get("protocol") or {}
    if provenance.get("sha256") != _sha256_file(protocol_path):
        raise RuntimeError("campaign validity gate protocol hash is stale")


def _run_job(
    *,
    root: Path,
    protocol_path: Path,
    task_manifest_path: Path,
    task_manifest_sha: str,
    kernelbench_dir: Path,
    phase: str,
    task_id: str,
    process_id: str,
    seed: int,
    candidates: list[dict[str, Any]],
    time_period: str = "screening",
    allow_failed_result: bool = False,
) -> dict[str, Any]:
    safe_task = hashlib.sha256(task_id.encode()).hexdigest()[:12]
    job_dir = root / safe_task / process_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / "job.json"
    output_path = job_dir / "result.json"
    job = {
        "protocol_path": str(protocol_path),
        "task_manifest_path": str(task_manifest_path),
        "task_manifest_sha256": task_manifest_sha,
        "kernelbench_dir": str(kernelbench_dir),
        "phase": phase,
        "task_id": task_id,
        "process_id": process_id,
        "time_period": time_period,
        "seed": seed,
        "candidates": candidates,
    }
    job_text = json.dumps(job, indent=2) + "\n"
    if job_path.exists() and job_path.read_text(encoding="utf-8") != job_text:
        raise RuntimeError(f"existing job changed: {job_path}")
    job_path.write_text(job_text, encoding="utf-8")
    if output_path.exists():
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if result.get("status") == "completed" or (
            allow_failed_result and result.get("status") == "failed"
        ):
            return result
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_holdout_worker.py",
            "--job",
            str(job_path),
            "--output",
            str(output_path),
        ],
        text=True,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    if completed.returncode or result.get("status") != "completed":
        if allow_failed_result and result.get("status") == "failed":
            return result
        raise RuntimeError(f"worker failed: {output_path}")
    return result


def _confirmation_process_indices(
    process_count: int,
    *,
    wave: str,
    time_periods: int,
) -> list[int]:
    if process_count <= 0 or time_periods <= 0:
        raise ValueError("confirmation process and time-period counts must be positive")
    if wave == "all":
        return list(range(process_count))
    if time_periods != 2 or wave not in {"wave1", "wave2"}:
        raise ValueError(f"unsupported confirmation wave: {wave}")
    split = (process_count + 1) // 2
    return list(range(0, split)) if wave == "wave1" else list(range(split, process_count))


def _confirmation_wave_for_index(
    process_index: int,
    *,
    process_count: int,
    time_periods: int,
) -> str:
    if process_index not in range(process_count):
        raise ValueError(f"process index out of range: {process_index}")
    if time_periods != 2:
        raise ValueError("only two confirmation time periods are supported")
    split = (process_count + 1) // 2
    return "wave1" if process_index < split else "wave2"


def _load_confirmation_records(
    confirmation_root: Path,
    winners: list[Any],
    *,
    process_count: int,
) -> tuple[list[TimingBlock], list[str]]:
    records: list[TimingBlock] = []
    missing: list[str] = []
    for winner in winners:
        safe_task = hashlib.sha256(winner.task_id.encode()).hexdigest()[:12]
        for process_index in range(process_count):
            process_id = f"p{process_index:02d}"
            output_path = confirmation_root / safe_task / process_id / "result.json"
            if not output_path.exists():
                missing.append(f"{winner.task_id}/{process_id}")
                continue
            result = json.loads(output_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed":
                missing.append(f"{winner.task_id}/{process_id}")
                continue
            records.extend(
                _timing_blocks_from_worker(
                    result,
                    phase="confirmation",
                    task_id=winner.task_id,
                )
            )
    return records, missing


def _write_wave1_integrity_lock(
    root: Path,
    confirmation_root: Path,
    winners: list[Any],
    *,
    selected_process_indices: list[int],
    minimum_separation_minutes: int,
) -> Path:
    fingerprints: list[dict[str, Any]] = []
    jobs: list[str] = []
    for winner in winners:
        safe_task = hashlib.sha256(winner.task_id.encode()).hexdigest()[:12]
        for process_index in selected_process_indices:
            process_id = f"p{process_index:02d}"
            output_path = confirmation_root / safe_task / process_id / "result.json"
            if not output_path.exists():
                raise RuntimeError(f"wave1 result missing before integrity lock: {output_path}")
            result = json.loads(output_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed":
                raise RuntimeError(f"wave1 result incomplete before integrity lock: {output_path}")
            fingerprints.append(_environment_fingerprint(result.get("environment") or {}))
            jobs.append(f"{winner.task_id}/{process_id}")
    unique = {json.dumps(item, sort_keys=True) for item in fingerprints}
    if len(unique) != 1:
        raise RuntimeError("wave1 workers did not preserve one GPU/software fingerprint")
    completed_at = datetime.now(timezone.utc)
    payload = {
        "schema_version": 1,
        "status": "LOCKED_AFTER_WAVE1_INTEGRITY_REVIEW_ONLY",
        "completed_at_utc": completed_at.isoformat(),
        "wave2_not_before_utc": datetime.fromtimestamp(
            completed_at.timestamp() + minimum_separation_minutes * 60,
            tz=timezone.utc,
        ).isoformat(),
        "minimum_separation_minutes": minimum_separation_minutes,
        "environment_fingerprint": fingerprints[0],
        "completed_jobs": jobs,
        "review_policy": "integrity_and_completeness_only_no_outcome_analysis",
    }
    path = root / "wave1_integrity_lock.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if (
            prior.get("environment_fingerprint") != payload["environment_fingerprint"]
            or prior.get("completed_jobs") != payload["completed_jobs"]
        ):
            raise RuntimeError("wave1 integrity lock already exists and is checksum-frozen")
        return path
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _validate_wave2_start(root: Path, temporal_rule: dict[str, Any]) -> None:
    path = root / "wave1_integrity_lock.json"
    if not path.exists():
        raise RuntimeError("wave2 requires the frozen wave1 integrity lock")
    lock = json.loads(path.read_text(encoding="utf-8"))
    not_before = datetime.fromisoformat(str(lock["wave2_not_before_utc"]))
    if datetime.now(timezone.utc) < not_before:
        raise RuntimeError(f"wave2 cannot start before {not_before.isoformat()}")
    current = _environment_fingerprint(cuda_environment_snapshot("cuda"))
    expected = lock.get("environment_fingerprint") or {}
    required_keys = ["gpu_uuid"] if temporal_rule["require_same_gpu_uuid"] else []
    if temporal_rule["require_same_software_fingerprint"]:
        required_keys.extend(["device_name", "driver_version", "torch_version", "torch_cuda_version"])
    mismatches = [key for key in required_keys if current.get(key) != expected.get(key)]
    if mismatches:
        raise RuntimeError(
            "wave2 environment differs from wave1 for: " + ", ".join(mismatches)
        )


def _environment_fingerprint(environment: dict[str, Any]) -> dict[str, Any]:
    return {
        key: environment.get(key)
        for key in (
            "gpu_uuid",
            "device_name",
            "driver_version",
            "torch_version",
            "torch_cuda_version",
            "allow_tf32_matmul",
            "allow_tf32_cudnn",
            "float32_matmul_precision",
        )
    }


def _screening_invalid_tasks(
    screening_root: Path,
    task_ids: list[str],
    *,
    winner_task_ids: set[str],
) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    for task_id in task_ids:
        if task_id in winner_task_ids:
            continue
        safe_task = hashlib.sha256(task_id.encode()).hexdigest()[:12]
        output_path = screening_root / safe_task / "screening" / "result.json"
        candidate_statuses: list[dict[str, Any]] = []
        worker_status = None
        worker_error = None
        if output_path.exists():
            result = json.loads(output_path.read_text(encoding="utf-8"))
            worker_status = result.get("status")
            worker_error = result.get("error")
            for candidate in result.get("candidate_results", []):
                verification = candidate.get("verification") or {}
                policy = candidate.get("policy") or {}
                candidate_statuses.append(
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "status": candidate.get("status"),
                        "policy_passed": policy.get("passed"),
                        "policy_rejection_reason": policy.get("rejection_reason"),
                        "verification_passed": verification.get("passed"),
                        "error": candidate.get("error") or verification.get("error"),
                    }
                )
        invalid.append(
            {
                "task_id": task_id,
                "reason": "no contract-valid, correct candidate produced screening timing",
                "worker_status": worker_status,
                "worker_error": worker_error,
                "candidate_statuses": candidate_statuses,
            }
        )
    return invalid


def _invalid_promotion_results(
    invalid_tasks: list[dict[str, Any]],
    *,
    practical_margin: float,
) -> list[PromotionResult]:
    return [
        PromotionResult(
            task_id=str(item["task_id"]),
            candidate_id="none",
            screening_speedup=None,
            confirmation_speedup=None,
            lower_speedup_bound=None,
            upper_speedup_bound=None,
            practical_margin=practical_margin,
            process_count=0,
            block_count=0,
            bootstrap_p_value=None,
            bh_adjusted_p_value=None,
            bh_rejected=False,
            label="INVALID",
            selection_optimism_log=None,
            notes=str(item["reason"]),
        )
        for item in invalid_tasks
    ]


def _timing_blocks_from_worker(
    result: dict[str, Any],
    *,
    phase: str,
    task_id: str,
) -> list[TimingBlock]:
    records: list[TimingBlock] = []
    for candidate in result.get("candidate_results", []):
        valid = candidate.get("status") == "completed"
        paired = candidate.get("paired_timing") or {}
        for block in paired.get("blocks", []):
            times = block["median_ms_per_launch"]
            records.append(
                TimingBlock(
                    phase=phase,
                    task_id=task_id,
                    candidate_id=candidate["candidate_id"],
                    process_id=str(paired.get("process_id")),
                    block_id=str(block["block_id"]),
                    eager_ms=float(times["eager"]),
                    candidate_ms=float(times["candidate"]),
                    correctness_passed=valid,
                    contract_passed=valid,
                )
            )
    return records


def _write_block_csv(path: Path, records: list[TimingBlock]) -> None:
    fieldnames = list(TimingBlock.__annotations__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _check_budget(root: Path, max_hours: float) -> None:
    elapsed = _recorded_worker_hours(root)
    if elapsed >= max_hours:
        raise RuntimeError(f"GPU time cap reached after {elapsed:.3f} recorded worker hours")


def _recorded_worker_hours(root: Path) -> float:
    elapsed_s = 0.0
    for path in root.glob("**/result.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            elapsed_s += max(0.0, float(data.get("elapsed_s", 0.0)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return elapsed_s / 3600.0


def _write_sha256_manifest(root: Path) -> Path:
    output = root / "SHA256SUMS"
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != output
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
