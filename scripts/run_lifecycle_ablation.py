#!/usr/bin/env python3
"""Measure the historical reconstruct-per-call confound outside primary timing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from openkernelforge.harness.inputs import clone_inputs
from openkernelforge.harness.paired_timing import (
    configure_precision_settings,
    cuda_environment_snapshot,
)
from openkernelforge.tasks.kernelbench_l1 import (
    KernelBenchModelReference,
    load_kernelbench_l1_tasks,
)
from openkernelforge.utils.env_probe import TRITON_EXECUTION_OK, probe_environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/workshop2026_holdout_protocol.yaml")
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--kernelbench-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="artifacts/workshop2026/lifecycle_ablation",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--task-id")
    parser.add_argument("--process-id")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    _require_cuda_linux()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if args.task_id is None or args.process_id is None or args.seed is None:
            raise ValueError("lifecycle worker requires task-id, process-id, and seed")
        return _worker(args, root)

    protocol_path = Path(args.protocol).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    manifest_path = Path(args.task_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kernelbench_dir = Path(args.kernelbench_dir).resolve()
    _validate_task_manifest(manifest, protocol, protocol_path, kernelbench_dir)
    lifecycle = protocol["controls"]["lifecycle"]
    task_ids = _select_lifecycle_task_ids(
        manifest,
        max_tasks=int(lifecycle["max_tasks"]),
    )
    processes = int(lifecycle["processes"])
    seed_base = int(lifecycle["process_seed_base"])
    paths: list[Path] = []
    for task_index, task_id in enumerate(task_ids):
        task_key = hashlib.sha256(task_id.encode()).hexdigest()[:12]
        for process_index in range(processes):
            process_id = f"p{process_index:02d}"
            output = root / task_key / process_id / "result.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() and json.loads(output.read_text()).get("status") == "completed":
                paths.append(output)
                continue
            completed = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--protocol",
                    str(protocol_path),
                    "--task-manifest",
                    str(manifest_path),
                    "--kernelbench-dir",
                    str(kernelbench_dir),
                    "--output-dir",
                    str(root),
                    "--task-id",
                    task_id,
                    "--process-id",
                    process_id,
                    "--seed",
                    str(seed_base + task_index * 10_007 + process_index * 101),
                ],
                text=True,
            )
            if completed.returncode or not output.exists():
                raise RuntimeError(f"lifecycle worker failed: {task_id}/{process_id}")
            paths.append(output)
    _summarize(root, paths, expected_tasks=len(task_ids), expected_processes=processes)
    _write_sha256_manifest(root)
    print(f"lifecycle ablation: {root}")
    return 0


def _worker(args, root: Path) -> int:
    started = time.monotonic()
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8")) or {}
    manifest_path = Path(args.task_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kernelbench_dir = Path(args.kernelbench_dir).resolve()
    _validate_task_manifest(
        manifest,
        protocol,
        Path(args.protocol).resolve(),
        kernelbench_dir,
    )
    if args.task_id not in manifest["selected_task_ids"]:
        raise RuntimeError(f"task is not in the frozen selection: {args.task_id}")
    precision = protocol["environment"]["precision"]
    precision_record = configure_precision_settings(
        allow_tf32_matmul=bool(precision["allow_tf32_matmul"]),
        allow_tf32_cudnn=bool(precision["allow_tf32_cudnn"]),
        float32_matmul_precision=str(precision["float32_matmul_precision"]),
    )
    lifecycle = protocol["controls"]["lifecycle"]
    blocks = int(lifecycle["blocks_per_process"])
    task = load_kernelbench_l1_tasks(
        kernelbench_dir,
        task_ids=[args.task_id],
        max_tasks=1,
    )[0]
    _validate_loaded_task_source(task, manifest, str(args.task_id))
    reference = task.reference_fn
    if not isinstance(reference, KernelBenchModelReference):
        raise RuntimeError("lifecycle ablation requires an official Model task")
    reference.prepare_for(torch.float32, torch.device("cuda"))
    rows: list[dict[str, Any]] = []
    rng = random.Random(int(args.seed))
    for block_index in range(blocks):
        inputs = task.generate_inputs(
            int(args.seed) + 1000 + block_index,
            task.benchmark_shapes[0],
            torch.float32,
            torch.device("cuda"),
        )
        method_order = ["persistent", "contaminated"]
        rng.shuffle(method_order)
        host_ms: dict[str, float] = {}
        event_ms: dict[str, float] = {}
        for method in method_order:
            fn = reference if method == "persistent" else reference.reconstruct_per_call
            host_ms[method] = _measure_synchronized_host_ms(fn, clone_inputs(inputs))
            event_ms[method] = _measure_enclosing_cuda_event_ms(fn, clone_inputs(inputs))
        _, decomposition = reference.reconstruct_per_call_profiled(*clone_inputs(inputs))
        rows.append(
            {
                "block_id": str(block_index),
                "input_seed": int(args.seed) + 1000 + block_index,
                "method_order": method_order,
                "host_ms": host_ms,
                "cuda_event_enclosing_call_ms": event_ms,
                "contaminated_decomposition": decomposition,
                "clock_snapshot": cuda_environment_snapshot("cuda"),
            }
        )
    task_key = hashlib.sha256(str(args.task_id).encode()).hexdigest()[:12]
    output = root / task_key / str(args.process_id) / "result.json"
    payload = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": args.task_id,
        "process_id": args.process_id,
        "seed": args.seed,
        "measurement_scope": "control_only_separate_from_primary_timing",
        "task_manifest_sha256": _sha256_file(manifest_path),
        "precision_settings": precision_record,
        "environment": cuda_environment_snapshot("cuda"),
        "blocks": rows,
        "elapsed_s": time.monotonic() - started,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def _measure_synchronized_host_ms(fn, inputs: tuple[Any, ...]) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        fn(*inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def _measure_enclosing_cuda_event_ms(fn, inputs: tuple[Any, ...]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        fn(*inputs)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _summarize(
    root: Path,
    paths: list[Path],
    *,
    expected_tasks: int,
    expected_processes: int,
) -> None:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        host_ratios: list[float] = []
        event_ratios: list[float] = []
        components: dict[str, list[float]] = {}
        for block in data["blocks"]:
            host = block["host_ms"]
            event = block["cuda_event_enclosing_call_ms"]
            host_ratios.append(float(host["contaminated"]) / float(host["persistent"]))
            event_ratios.append(float(event["contaminated"]) / float(event["persistent"]))
            for name, value in block["contaminated_decomposition"].items():
                components.setdefault(name, []).append(float(value))
        row = {
            "task_id": data["task_id"],
            "process_id": data["process_id"],
            "blocks": len(data["blocks"]),
            "median_host_lifecycle_inflation": statistics.median(host_ratios),
            "median_enclosing_event_inflation": statistics.median(event_ratios),
        }
        row.update({f"median_{name}": statistics.median(values) for name, values in components.items()})
        rows.append(row)
    csv_path = root / "lifecycle_ablation.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    complete = (
        len({row["task_id"] for row in rows}) == expected_tasks
        and all(
            sum(row["task_id"] == task_id for row in rows) == expected_processes
            for task_id in {row["task_id"] for row in rows}
        )
    )
    summary = {
        "status": "PASS" if complete else "FAIL",
        "expected_tasks": expected_tasks,
        "expected_processes_per_task": expected_processes,
        "completed_process_rows": len(rows),
        "median_host_lifecycle_inflation": (
            statistics.median(float(row["median_host_lifecycle_inflation"]) for row in rows)
            if rows
            else None
        ),
        "median_enclosing_event_inflation": (
            statistics.median(float(row["median_enclosing_event_inflation"]) for row in rows)
            if rows
            else None
        ),
        "caveat": (
            "enclosing CUDA events bracket host reconstruction and therefore are not "
            "interpreted as pure device compute; synchronized host latency is primary"
        ),
    }
    (root / "lifecycle_ablation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "lifecycle_ablation.md").write_text(
        "\n".join(
            [
                "# Isolated Reference Lifecycle Ablation",
                "",
                "This control runs in disposable processes outside candidate screening.",
                "Synchronized host latency captures the complete historical call; enclosing "
                "CUDA events are reported separately and are not treated as pure device time.",
                "",
                f"- status: {summary['status']}",
                f"- completed process rows: {summary['completed_process_rows']}",
                "- median host lifecycle inflation: "
                f"{summary['median_host_lifecycle_inflation']}",
                "- median enclosing-event inflation: "
                f"{summary['median_enclosing_event_inflation']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _require_cuda_linux() -> None:
    environment = probe_environment()
    if platform.system() == "Darwin" or environment.viability != TRITON_EXECUTION_OK:
        raise RuntimeError("lifecycle ablation requires a Linux CUDA/Triton environment")


def _validate_task_manifest(
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    protocol_path: Path,
    kernelbench_dir: Path,
) -> None:
    if manifest.get("status") != "FROZEN_BEFORE_CANDIDATE_PERFORMANCE":
        raise RuntimeError("lifecycle ablation requires a frozen task manifest")
    if manifest.get("study_id") != protocol.get("study", {}).get("id"):
        raise RuntimeError("task manifest study does not match lifecycle protocol")
    expected_commit = str(protocol.get("kernelbench", {}).get("commit") or "")
    actual_commit = subprocess.run(
        ["git", "-C", str(kernelbench_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        expected_commit
        and (
            manifest.get("kernelbench_commit") != expected_commit
            or actual_commit != expected_commit
        )
    ):
        raise RuntimeError("task manifest KernelBench commit does not match protocol")
    expected_protocol_sha = _sha256_file(protocol_path)
    if manifest.get("protocol_sha256") != expected_protocol_sha:
        raise RuntimeError("task manifest protocol hash does not match current protocol")


def _validate_loaded_task_source(
    task: Any,
    manifest: dict[str, Any],
    task_id: str,
) -> None:
    rows = [row for row in manifest.get("rows", []) if str(row.get("task_id")) == task_id]
    if len(rows) != 1:
        raise RuntimeError("frozen lifecycle task source row is missing or duplicated")
    source_path = Path(str(task.metadata["source_path"])).resolve()
    if _sha256_file(source_path) != rows[0].get("source_sha256"):
        raise RuntimeError("lifecycle task source differs from frozen manifest")


def _select_lifecycle_task_ids(
    manifest: dict[str, Any],
    *,
    max_tasks: int,
) -> list[str]:
    if max_tasks <= 0:
        raise ValueError("lifecycle max_tasks must be positive")
    selected_ids = [str(item) for item in manifest["selected_task_ids"]]
    selected_set = set(selected_ids)
    families: dict[str, str] = {
        str(row["task_id"]): str(row.get("family", "kernelbench_l1"))
        for row in manifest.get("rows", [])
        if str(row.get("task_id")) in selected_set
    }
    chosen: list[str] = []
    seen_families: set[str] = set()
    for task_id in selected_ids:
        family = families.get(task_id, "kernelbench_l1")
        if family not in seen_families:
            chosen.append(task_id)
            seen_families.add(family)
            if len(chosen) == max_tasks:
                return chosen
    for task_id in selected_ids:
        if task_id not in chosen:
            chosen.append(task_id)
            if len(chosen) == max_tasks:
                break
    return chosen


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sha256_manifest(root: Path) -> Path:
    output = root / "SHA256SUMS"
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != output
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    raise SystemExit(main())
