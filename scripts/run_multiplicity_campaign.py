#!/usr/bin/env python3
"""Run the separate all-candidate controlled multiplicity experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from openkernelforge.reports.holdout_confirmation import TimingBlock
from openkernelforge.reports.selection_multiplicity import (
    analyze_selection_multiplicity,
    write_multiplicity_csv,
)
from openkernelforge.utils.env_probe import TRITON_EXECUTION_OK, probe_environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/workshop2026_multiplicity_protocol.yaml",
    )
    parser.add_argument(
        "--candidate-manifest",
        default="artifacts/workshop2026/multiplicity/candidate_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/workshop2026/multiplicity/campaign",
    )
    parser.add_argument("--max-gpu-hours", type=float, default=5.0)
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()
    _require_cuda_linux()
    protocol_path = Path(args.protocol).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    manifest_path = Path(args.candidate_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest, manifest_path=manifest_path, protocol_path=protocol_path)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_sha = _sha256_file(manifest_path)
    task_ids = [str(item) for item in protocol["tasks"]["ids"]]

    for task_index, task_id in enumerate(task_ids):
        _check_budget(root, args.max_gpu_hours)
        _run_job(
            root=root / "screening",
            protocol_path=protocol_path,
            manifest_path=manifest_path,
            manifest_sha=manifest_sha,
            phase="screening",
            task_id=task_id,
            process_id="screening",
            seed=int(protocol["screening"]["process_seed"]) + task_index * 101,
            candidates=manifest["tasks"][task_id],
        )
    if args.screen_only:
        _write_sha256_manifest(root)
        print(f"multiplicity screening complete: {root}")
        return 0

    process_count = int(protocol["confirmation"]["fresh_processes"])
    seed_base = int(protocol["confirmation"]["process_seed_base"])
    for task_index, task_id in enumerate(task_ids):
        for process_index in range(process_count):
            _check_budget(root, args.max_gpu_hours)
            _run_job(
                root=root / "confirmation",
                protocol_path=protocol_path,
                manifest_path=manifest_path,
                manifest_sha=manifest_sha,
                phase="confirmation",
                task_id=task_id,
                process_id=f"p{process_index:02d}",
                seed=seed_base + task_index * 10_007 + process_index * 101,
                candidates=manifest["tasks"][task_id],
            )

    records = _collect_records(root, task_ids, process_count=process_count)
    blocks_path = root / "all_candidate_timing_blocks.csv"
    _write_blocks(blocks_path, records)
    analysis = protocol["analysis"]
    rows = analyze_selection_multiplicity(
        records,
        budgets=[int(item) for item in analysis["candidate_budgets"]],
        resamples=int(analysis["resamples"]),
        seed=int(analysis["seed"]),
        practical_margin=float(analysis["practical_speedup_margin"]),
        required_confirmation_processes=process_count,
    )
    write_multiplicity_csv(root / "selection_multiplicity.csv", rows)
    _write_summary(root / "selection_multiplicity.md", rows)
    _write_sha256_manifest(root)
    print(f"multiplicity campaign complete: {root}")
    return 0


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    protocol_path: Path,
) -> None:
    if manifest.get("status") != "FROZEN_BEFORE_ANY_TIMING":
        raise RuntimeError("multiplicity candidate manifest is not frozen")
    if manifest.get("protocol_sha256") != _sha256_file(protocol_path):
        raise RuntimeError("multiplicity protocol changed after candidate freeze")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    expected = int(protocol["candidates"]["variants_per_task"])
    for task_id in protocol["tasks"]["ids"]:
        candidates = manifest.get("tasks", {}).get(task_id, [])
        if len(candidates) != expected:
            raise RuntimeError(f"{task_id} has {len(candidates)} candidates; expected {expected}")
        for candidate in candidates:
            path = Path(candidate["path"]).resolve()
            metadata_path = Path(candidate["metadata_path"]).resolve()
            if _sha256_file(path) != candidate["sha256"]:
                raise RuntimeError(f"candidate source checksum mismatch: {task_id}")
            if _sha256_file(metadata_path) != candidate["metadata_sha256"]:
                raise RuntimeError(f"candidate metadata checksum mismatch: {task_id}")
    checksum_path = manifest_path.with_suffix(".sha256")
    if checksum_path.exists():
        recorded = checksum_path.read_text(encoding="utf-8").split()[0]
        if recorded != _sha256_file(manifest_path):
            raise RuntimeError("multiplicity manifest checksum file mismatch")


def _run_job(
    *,
    root: Path,
    protocol_path: Path,
    manifest_path: Path,
    manifest_sha: str,
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
        "candidate_manifest_sha256": manifest_sha,
        "phase": phase,
        "task_id": task_id,
        "process_id": process_id,
        "seed": seed,
        "candidates": candidates,
    }
    text = json.dumps(job, indent=2) + "\n"
    if job_path.exists() and job_path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"existing multiplicity job changed: {job_path}")
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
        text=True,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if completed.returncode or result.get("status") != "completed":
        raise RuntimeError(f"multiplicity worker failed: {result_path}")
    return result


def _collect_records(root: Path, task_ids: list[str], *, process_count: int) -> list[TimingBlock]:
    records: list[TimingBlock] = []
    jobs = [("screening", "screening")]
    jobs.extend(("confirmation", f"p{index:02d}") for index in range(process_count))
    for task_id in task_ids:
        for phase, process_id in jobs:
            path = root / phase / task_id / process_id / "result.json"
            if not path.exists():
                raise RuntimeError(f"missing multiplicity worker result: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))
            for candidate in data.get("candidate_results", []):
                valid = candidate.get("status") == "completed"
                paired = candidate.get("paired_timing") or {}
                for block in paired.get("blocks", []):
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


def _write_blocks(path: Path, records: list[TimingBlock]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TimingBlock.__annotations__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _write_summary(path: Path, rows) -> None:
    lines = [
        "# Controlled Candidate-Budget Multiplicity",
        "",
        "Every candidate has screening and independent fresh-process confirmation data.",
        "",
        "| K | Task resamples | Apparent win rate | Confirmed win rate | Median optimism | Task-bootstrap 95% interval |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.candidate_budget} | {row.task_resamples} | "
            f"{row.apparent_win_rate:.4f} | {row.confirmed_win_rate:.4f} | "
            f"{row.median_selection_optimism_log:.6f} | "
            f"[{row.selection_optimism_log_ci_lower:.6f}, "
            f"{row.selection_optimism_log_ci_upper:.6f}] |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recorded_worker_hours(root: Path) -> float:
    total = 0.0
    for path in root.glob("**/result.json"):
        try:
            total += max(0.0, float(json.loads(path.read_text())["elapsed_s"]))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return total / 3600.0


def _check_budget(root: Path, max_hours: float) -> None:
    used = _recorded_worker_hours(root)
    if used >= max_hours:
        raise RuntimeError(f"GPU time cap reached after {used:.3f} recorded worker hours")


def _require_cuda_linux() -> None:
    environment = probe_environment()
    if platform.system() == "Darwin" or environment.viability != TRITON_EXECUTION_OK:
        raise RuntimeError("multiplicity campaign requires a Linux CUDA/Triton environment")


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
