#!/usr/bin/env python3
"""Fresh-process confirmation of one frozen candidate against torch.compile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK = "KernelBench__level1__12_Matmul_with_diagonal_matrices_"
DEFAULT_CANDIDATE = "candidate_000"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernelbench-dir", required=True)
    parser.add_argument(
        "--protocol", default="configs/workshop2026_holdout_protocol.yaml"
    )
    parser.add_argument(
        "--task-manifest",
        default="artifacts/workshop2026/task_selection_manifest.json",
    )
    parser.add_argument(
        "--candidate-manifest",
        default="artifacts/workshop2026/candidate_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/workshop2026/compiler_confirmation_a4500",
    )
    parser.add_argument("--task-id", default=DEFAULT_TASK)
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE)
    parser.add_argument("--processes", type=int, default=7)
    parser.add_argument("--seed-base", type=int, default=401_071)
    args = parser.parse_args()

    if args.processes <= 0:
        raise ValueError("processes must be positive")
    protocol = _resolve(args.protocol)
    task_manifest_path = _resolve(args.task_manifest)
    candidate_manifest_path = _resolve(args.candidate_manifest)
    output_dir = _resolve(args.output_dir)
    kernelbench_dir = Path(args.kernelbench_dir).resolve()
    task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if args.task_id not in task_manifest.get("selected_task_ids", []):
        raise RuntimeError("requested task is absent from the frozen task manifest")
    candidates = [
        row
        for row in candidate_manifest.get("tasks", {}).get(args.task_id, [])
        if row.get("candidate_id") == args.candidate_id
    ]
    if len(candidates) != 1:
        raise RuntimeError("requested candidate is absent or duplicated")

    output_dir.mkdir(parents=True, exist_ok=True)
    task_manifest_sha = _sha256_file(task_manifest_path)
    result_paths: list[Path] = []
    for index in range(args.processes):
        process_id = f"compiler_p{index:02d}"
        process_dir = output_dir / "processes" / process_id
        process_dir.mkdir(parents=True, exist_ok=True)
        job_path = process_dir / "job.json"
        result_path = process_dir / "result.json"
        job = {
            "protocol_path": str(protocol),
            "task_manifest_path": str(task_manifest_path),
            "task_manifest_sha256": task_manifest_sha,
            "kernelbench_dir": str(kernelbench_dir),
            # The original screening block includes a materialized compile baseline.
            "phase": "screening",
            "validation_role": "fresh_process_compiler_confirmation",
            "task_id": args.task_id,
            "process_id": process_id,
            "seed": args.seed_base + index * 101,
            "candidates": candidates,
        }
        serialized = json.dumps(job, indent=2) + "\n"
        if job_path.exists() and job_path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"existing compiler confirmation job changed: {job_path}")
        job_path.write_text(serialized, encoding="utf-8")
        if not _completed(result_path):
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/benchmark_holdout_worker.py",
                    "--job",
                    str(job_path),
                    "--output",
                    str(result_path),
                ]
            )
            if completed.returncode:
                raise RuntimeError(f"compiler confirmation worker failed: {result_path}")
        result_paths.append(result_path)

    rows = analyze_results(result_paths)
    csv_path = output_dir / "compiler_confirmation.csv"
    _write_csv(csv_path, rows)
    summary = summarize_results(rows)
    summary.update(
        {
            "task_id": args.task_id,
            "candidate_id": args.candidate_id,
            "candidate_sha256": candidates[0]["sha256"],
            "processes": args.processes,
            "prespecified_margin": 0.02,
            "validation_role": "separate_cross_gpu_compiler_rung_confirmation",
        }
    )
    summary_path = output_dir / "compiler_confirmation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = output_dir / "compiler_confirmation.md"
    _write_report(report_path, summary, rows)
    _write_sha256_manifest(output_dir)
    print(f"compiler confirmation summary: {summary_path}")
    print(f"compiler confirmation report: {report_path}")
    return 0


def analyze_results(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise RuntimeError(f"worker did not complete: {path}")
        candidates = payload.get("candidate_results", [])
        if len(candidates) != 1 or candidates[0].get("status") != "completed":
            raise RuntimeError(f"candidate did not complete: {path}")
        blocks = candidates[0]["paired_timing"]["blocks"]
        ratios = [
            float(block["median_ms_per_launch"]["compile"])
            / float(block["median_ms_per_launch"]["candidate"])
            for block in blocks
        ]
        environment = payload.get("environment", {})
        rows.append(
            {
                "process_id": payload["job"]["process_id"],
                "seed": payload["job"]["seed"],
                "blocks": len(blocks),
                "candidate_vs_compile_median": statistics.median(ratios),
                "candidate_vs_compile_min": min(ratios),
                "candidate_vs_compile_max": max(ratios),
                "compile_and_first_call_ms": payload["torch_compile"][
                    "compile_and_first_call_ms"
                ],
                "gpu": environment.get("device_name", "not recorded"),
                "driver": environment.get("driver_version", "not recorded"),
                "result_path": path.as_posix(),
            }
        )
    return rows


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    process_speedups = [float(row["candidate_vs_compile_median"]) for row in rows]
    median_speedup = statistics.median(process_speedups)
    return {
        "status": "completed",
        "median_candidate_vs_compile": median_speedup,
        "per_process_candidate_vs_compile": process_speedups,
        "above_compile_parity": median_speedup > 1.0,
        "above_compile_1_02": median_speedup > 1.02,
        "interpretation": (
            "confirmed_above_compile_margin"
            if median_speedup > 1.02
            else "not_confirmed_above_compile_margin"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    lines = [
        "# Fresh-Process Compiler-Rung Confirmation",
        "",
        "This validation reuses one frozen generated candidate and does not alter the primary candidate-versus-eager campaign.",
        "",
        f"- Task: `{summary['task_id']}`.",
        f"- Candidate: `{summary['candidate_id']}`.",
        f"- Fresh processes: {summary['processes']}.",
        f"- Median candidate speedup versus compile: {summary['median_candidate_vs_compile']:.6f}x.",
        f"- Result: `{summary['interpretation']}`.",
        "",
        "| Process | Blocks | Candidate vs compile | Compile + first call (ms) | GPU |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['process_id']} | {row['blocks']} | "
            f"{float(row['candidate_vs_compile_median']):.6f}x | "
            f"{float(row['compile_and_first_call_ms']):.1f} | {row['gpu']} |"
        )
    lines.extend(
        [
            "",
            "The runtime comparison excludes compilation. Compile-and-first-call latency is retained as deployment context rather than folded into the speedup.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def _write_sha256_manifest(root: Path) -> Path:
    output = root / "SHA256SUMS"
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != output
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
