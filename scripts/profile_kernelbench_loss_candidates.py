from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openkernelforge.harness.policy import CandidatePolicyResult, check_candidate_policy
from openkernelforge.harness.sandbox import load_candidate_from_path


ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOT = ROOT / "artifacts" / "runpod_imports"
SUMMARY = ROOT / "reports" / "profiling" / "kernelbench_loss_profiler_summary.md"
OPS_CSV = ROOT / "reports" / "tables" / "kernelbench_loss_profiler_ops.csv"
MEMORY_CSV = ROOT / "reports" / "tables" / "kernelbench_loss_profiler_memory.csv"
MECHANISM_CSV = ROOT / "reports" / "tables" / "kernelbench_loss_mechanism_summary.csv"
HISTORICAL_RUN_IDS = {"20260520_202314", "20260520_213128"}
CURRENT_POLICY_VERSION = CandidatePolicyResult(passed=False).policy_version


@dataclass(frozen=True)
class Target:
    task: str
    task_id: str
    run_id: str
    existing_speedup_vs_eager: str
    existing_speedup_vs_compile: str


TARGETS = [
    Target(
        task="CrossEntropyLoss",
        task_id="KernelBench__level1__95_CrossEntropyLoss",
        run_id="20260520_202314",
        existing_speedup_vs_eager="1.992x",
        existing_speedup_vs_compile="2.895x",
    ),
    Target(
        task="TripletMarginLoss",
        task_id="KernelBench__level1__99_TripletMarginLoss",
        run_id="20260520_202314",
        existing_speedup_vs_eager="4.176x",
        existing_speedup_vs_compile="3.208x",
    ),
    Target(
        task="KLDivLoss",
        task_id="KernelBench__level1__98_KLDivLoss",
        run_id="20260520_213128",
        existing_speedup_vs_eager="1.843x",
        existing_speedup_vs_compile="1.028x",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernelbench-dir",
        default=os.environ.get("KERNELBENCH_DIR") or _first_existing_kernelbench_dir(),
        help="Local KernelBench checkout. Required only for CUDA profiler execution.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument(
        "--one-shot-run-dir",
        type=Path,
        help="Corrected KernelBench run directory containing CE and Triplet records.",
    )
    parser.add_argument(
        "--repair-run-dir",
        type=Path,
        help="Corrected KernelBench repair run directory containing the KLDiv record.",
    )
    parser.add_argument(
        "--allow-historical-debug",
        action="store_true",
        help=(
            "Permit execution of policy-clean sources from the historically affected runs for "
            "debugging only. These rows remain ineligible as paper evidence."
        ),
    )
    args = parser.parse_args()

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OPS_CSV.parent.mkdir(parents=True, exist_ok=True)

    records = [
        _target_record(
            target,
            run_dir=(
                args.repair_run_dir
                if target.task == "KLDivLoss" and args.repair_run_dir
                else args.one_shot_run_dir
                if target.task != "KLDivLoss" and args.one_shot_run_dir
                else IMPORT_ROOT / "runs" / target.run_id
            ),
        )
        for target in TARGETS
    ]
    cuda_available, cuda_reason = _cuda_status()
    kernelbench_dir = Path(args.kernelbench_dir).expanduser() if args.kernelbench_dir else None
    can_profile = cuda_available and kernelbench_dir is not None and kernelbench_dir.exists()

    op_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    status_lines: list[str] = []

    if not cuda_available:
        status_lines.append(f"Profiler skipped: {cuda_reason}.")
    elif not can_profile:
        status_lines.append(f"Profiler skipped: KernelBench checkout not found at {kernelbench_dir}.")
    else:
        status_lines.append(f"Profiler enabled on CUDA using KernelBench checkout: {kernelbench_dir}.")

    for record in records:
        static = _static_mechanism(record)
        policy = _policy_result(record)
        record["current_policy_passed"] = policy.passed if policy is not None else False
        record["current_policy_reason"] = (
            policy.rejection_reason if policy is not None and not policy.passed else "passed"
        )
        if policy is None or not policy.passed:
            static["confidence"] = "historical source only"
            static["caveat"] = (
                "historical source is excluded from attribution; current policy result: "
                f"{record['current_policy_reason']}"
            )
        profiler_status = "skipped"
        reason = _execution_block_reason(
            record,
            can_profile=can_profile,
            environment_reason=status_lines[0],
            allow_historical_debug=args.allow_historical_debug,
        )
        if not record["candidate_exists"]:
            reason = "preserved candidate source missing"
        if reason:
            op_rows.extend(_skipped_op_rows(record, reason))
            memory_rows.extend(_skipped_memory_rows(record, reason))
            mechanism_rows.append(_mechanism_row(record, static, profiler_status))
            continue
        try:
            profiled_ops, profiled_memory = _profile_record(
                record,
                kernelbench_dir=kernelbench_dir,
                warmup=args.warmup,
                top_k=args.top_k,
                include_compile=not args.skip_compile,
            )
            op_rows.extend(profiled_ops)
            memory_rows.extend(profiled_memory)
            profiler_status = (
                "historical_debug_profiled" if record["historical_run"] else "profiled"
            )
        except Exception as exc:  # pragma: no cover - depends on CUDA/KernelBench
            reason = f"profile failed: {type(exc).__name__}: {exc}"
            op_rows.extend(_skipped_op_rows(record, reason))
            memory_rows.extend(_skipped_memory_rows(record, reason))
        mechanism_rows.append(_mechanism_row(record, static, profiler_status))

    _write_csv(OPS_CSV, op_rows)
    _write_csv(MEMORY_CSV, memory_rows)
    _write_csv(MECHANISM_CSV, mechanism_rows)
    SUMMARY.write_text(_summary_text(records, status_lines, mechanism_rows, op_rows), encoding="utf-8")
    print(f"Wrote {SUMMARY}")
    print(f"Wrote {OPS_CSV}")
    print(f"Wrote {MEMORY_CSV}")
    print(f"Wrote {MECHANISM_CSV}")
    return 0


def _target_record(target: Target, *, run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    row = _load_result_row(run_dir / "results.jsonl", target.task_id)
    candidate_path = _mapped_path(row.get("candidate_path"), run_dir=run_dir) if row else None
    source_path = Path(str(row.get("kernelbench_source_path"))) if row else None
    verification = row.get("verification", {}) if row else {}
    cases = verification.get("cases") or []
    first_case = cases[0] if cases else {}
    benchmark = row.get("benchmark", {}) if row else {}
    summary = row.get("benchmark_summary", {}) if row else {}
    return {
        "task": target.task,
        "task_id": target.task_id,
        "run_id": target.run_id,
        "resolved_run_dir": str(run_dir),
        "historical_run": run_dir.name in HISTORICAL_RUN_IDS,
        "contract_recorded": bool(
            row
            and row.get("policy_version") == CURRENT_POLICY_VERSION
            and row.get("candidate_contract")
        ),
        "recorded_policy_version": row.get("policy_version", "not available") if row else "not available",
        "candidate_contract": row.get("candidate_contract", "not available") if row else "not available",
        "result_path": str(run_dir / "results.jsonl"),
        "candidate_path": str(candidate_path) if candidate_path else "not available",
        "candidate_exists": bool(candidate_path and candidate_path.exists()),
        "kernelbench_source_path": str(source_path) if source_path else "not available",
        "source_exists": bool(source_path and source_path.exists()),
        "label": row.get("candidate_label", "not available") if row else "not available",
        "speedup_vs_eager": _format_speedup(summary.get("speedup_vs_eager"), target.existing_speedup_vs_eager),
        "speedup_vs_compile": _format_speedup(summary.get("speedup_vs_compile"), target.existing_speedup_vs_compile),
        "input_shape": str(benchmark.get("shape") or first_case.get("shape") or "not available"),
        "dtype": str(benchmark.get("dtype") or "float32"),
        "tolerance": _tolerance_summary(first_case),
        "verified": bool(row.get("verification_passed")) if row else False,
        "benchmarked": bool(row.get("benchmarked")) if row else False,
    }


def _load_result_row(path: Path, task_id: str) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task_id") == task_id:
            if row.get("verification_passed") or task_id.endswith("98_KLDivLoss"):
                return row
    raise FileNotFoundError(f"result row not found for {task_id} in {path}")


def _mapped_path(path_text: str | None, *, run_dir: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [run_dir / path, IMPORT_ROOT / path, ROOT / path]
    return next((candidate for candidate in candidates if candidate.exists()), ROOT / path)


def _policy_result(record: dict[str, Any]):
    candidate_path = Path(record["candidate_path"])
    if not candidate_path.exists():
        return None
    source = candidate_path.read_text(encoding="utf-8", errors="replace")
    return check_candidate_policy(
        source,
        allow_torch_fallback=False,
        require_triton=True,
    )


def _execution_block_reason(
    record: dict[str, Any],
    *,
    can_profile: bool,
    environment_reason: str,
    allow_historical_debug: bool,
) -> str | None:
    if not record["candidate_exists"]:
        return "preserved candidate source missing"
    if not record["current_policy_passed"]:
        return f"current policy rejected candidate: {record['current_policy_reason']}"
    if record["historical_run"] and not allow_historical_debug:
        return (
            "historical adapter output is blocked from execution by default; use a corrected run "
            "directory or --allow-historical-debug"
        )
    if not record["historical_run"] and not record["contract_recorded"]:
        return (
            "run record lacks corrected candidate-contract and "
            f"{CURRENT_POLICY_VERSION} policy metadata"
        )
    if not can_profile:
        return environment_reason
    return None


def _format_speedup(value: Any, fallback: str) -> str:
    try:
        return f"{float(value):.3f}x"
    except (TypeError, ValueError):
        return fallback


def _tolerance_summary(first_case: dict[str, Any]) -> str:
    message = first_case.get("message")
    if message and "rtol=" in message and "atol=" in message:
        return message.rsplit(" with ", 1)[-1]
    return "rtol=1e-4, atol=1e-5"


def _cuda_status() -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"PyTorch import failed: {type(exc).__name__}: {exc}"
    if not torch.cuda.is_available():
        return False, "CUDA is not available in this workspace"
    return True, f"CUDA available: {torch.cuda.get_device_name(0)}"


def _first_existing_kernelbench_dir() -> str | None:
    candidates = [
        Path("/workspace/KernelBench"),
        Path("/workspace/kernelbench"),
        ROOT / "KernelBench",
        ROOT / "kernelbench",
        ROOT.parent / "KernelBench",
        ROOT.parent / "kernelbench",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _static_mechanism(record: dict[str, Any]) -> dict[str, str]:
    text = ""
    candidate_path = Path(record["candidate_path"])
    if candidate_path.exists():
        text = candidate_path.read_text(encoding="utf-8", errors="replace")
    task = record["task"]
    if task == "CrossEntropyLoss":
        pattern = "row-wise log-sum-exp, target gather, per-row loss buffer, final mean reduction"
        eager = "likely PyTorch cross-entropy decomposition/library loss path over logits and targets"
        confidence = "medium"
        caveat = "profiler unavailable locally; static source shows fused row-wise loss but final mean remains a separate reduction"
    elif task == "TripletMarginLoss":
        pattern = "single Triton kernel computes positive/negative distances, sqrt, hinge, and per-row loss before mean"
        eager = "likely sequence of subtractions, square/pow, reductions, sqrt, margin/hinge, and final reduction"
        confidence = "medium-high"
        caveat = "profiler unavailable locally; source is consistent with avoiding many intermediate tensors"
    elif task == "KLDivLoss":
        pattern = "torch.log on predictions plus Triton KL elementwise term and torch.sum batchmean reduction"
        eager = "likely log transform, KL elementwise computation, and batchmean reduction"
        confidence = "medium"
        caveat = "mixed attribution: repaired candidate still uses torch.log and torch.sum outside Triton"
    else:
        pattern = "not available"
        eager = "not available"
        confidence = "low"
        caveat = "unknown task"
    if text and "torch.log" not in text and task == "KLDivLoss":
        caveat = "static source unavailable or unexpected; verify candidate before attributing mechanism"
    return {
        "dominant_eager_costs": eager,
        "candidate_pattern": pattern,
        "confidence": confidence,
        "caveat": caveat,
    }


def _mechanism_row(record: dict[str, Any], static: dict[str, str], profiler_status: str) -> dict[str, Any]:
    return {
        "task": record["task"],
        "existing_speedup_vs_eager": record["speedup_vs_eager"],
        "existing_speedup_vs_compile": record["speedup_vs_compile"],
        "dominant_eager_costs": static["dominant_eager_costs"],
        "candidate_pattern": static["candidate_pattern"],
        "attribution_confidence": static["confidence"],
        "profiler_status": profiler_status,
        "caveat": static["caveat"],
        "label": record["label"],
        "input_shape": record["input_shape"],
        "dtype": record["dtype"],
        "candidate_path": record["candidate_path"],
        "kernelbench_source_path": record["kernelbench_source_path"],
        "evidence_status": (
            "historical_debug_only"
            if record["historical_run"] and profiler_status == "historical_debug_profiled"
            else "historical_adapter_output_only"
            if record["historical_run"]
            else "corrected_run_diagnostic"
        ),
        "current_policy": record["current_policy_reason"],
    }


def _skipped_op_rows(record: dict[str, Any], reason: str) -> list[dict[str, Any]]:
    return [
        {
            "task": record["task"],
            "path": path,
            "status": "skipped",
            "operator": "not available",
            "self_cpu_time_us": "not available",
            "cpu_time_total_us": "not available",
            "self_cuda_time_us": "not available",
            "cuda_time_total_us": "not available",
            "calls": "not available",
            "reason": reason,
        }
        for path in ("eager", "candidate", "compile")
    ]


def _skipped_memory_rows(record: dict[str, Any], reason: str) -> list[dict[str, Any]]:
    return [
        {
            "task": record["task"],
            "path": path,
            "status": "skipped",
            "cpu_memory_bytes": "not available",
            "cuda_memory_bytes": "not available",
            "reason": reason,
        }
        for path in ("eager", "candidate", "compile")
    ]


def _profile_record(
    record: dict[str, Any],
    *,
    kernelbench_dir: Path,
    warmup: int,
    top_k: int,
    include_compile: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:  # pragma: no cover - CUDA only
    import torch

    from openkernelforge.tasks.kernelbench_l1 import (
        bind_kernelbench_candidate,
        load_kernelbench_l1_tasks,
    )

    policy = _policy_result(record)
    if policy is None or not policy.passed:
        reason = policy.rejection_reason if policy is not None else "candidate missing"
        raise ValueError(f"candidate failed current policy before import: {reason}")
    task = load_kernelbench_l1_tasks(kernelbench_dir, task_ids=[record["task_id"]], max_tasks=1)[0]
    candidate_module = load_candidate_from_path(
        Path(record["candidate_path"]), require_forward=False
    ).module
    device = torch.device("cuda")
    dtype = getattr(torch, record["dtype"], torch.float32)
    shape = tuple(int(x) for x in json.loads(record["input_shape"].replace("'", '"')))
    inputs = task.input_generator(0, shape, dtype, device)
    prepare_reference = getattr(task.reference_fn, "prepare_for", None)
    reference_callable = (
        prepare_reference(dtype, device) if callable(prepare_reference) else task.reference_fn
    )
    candidate_callable = bind_kernelbench_candidate(
        task,
        candidate_module,
        dtype=dtype,
        device=device,
    )
    paths: list[tuple[str, Callable[..., Any]]] = [
        ("eager", reference_callable),
        ("candidate", candidate_callable),
    ]
    if include_compile and hasattr(torch, "compile"):
        try:
            compiled = torch.compile(reference_callable, mode="max-autotune")
            compiled(*inputs)
            torch.cuda.synchronize()
            paths.append(("compile", compiled))
        except Exception:
            pass

    op_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    for path_name, fn in paths:
        for _ in range(max(0, warmup)):
            fn(*inputs)
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            fn(*inputs)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        allocated_after = torch.cuda.memory_allocated()
        key_averages = prof.key_averages()
        sorted_events = sorted(
            key_averages,
            key=lambda item: _event_time(item, self_time=True)
            or getattr(item, "self_cpu_time_total", 0.0),
            reverse=True,
        )
        for event in sorted_events[:top_k]:
            op_rows.append(
                {
                    "task": record["task"],
                    "path": path_name,
                    "status": "profiled",
                    "operator": event.key,
                    "self_cpu_time_us": f"{getattr(event, 'self_cpu_time_total', 0.0):.3f}",
                    "cpu_time_total_us": f"{getattr(event, 'cpu_time_total', 0.0):.3f}",
                    "self_cuda_time_us": f"{_event_time(event, self_time=True):.3f}",
                    "cuda_time_total_us": f"{_event_time(event, self_time=False):.3f}",
                    "calls": getattr(event, "count", "not available"),
                    "reason": "profiler diagnostic only; not a benchmark result",
                }
            )
        memory_rows.append(
            {
                "task": record["task"],
                "path": path_name,
                "status": "profiled",
                "cpu_memory_bytes": "see profiler operator rows",
                "cuda_memory_bytes": str(int(max(0, peak - allocated_before))),
                "reason": (
                    "incremental peak over pre-call torch.cuda.memory_allocated; "
                    f"before={allocated_before}, after={allocated_after}, absolute_peak={peak}"
                ),
            }
        )
    return op_rows, memory_rows


def _event_time(event: Any, *, self_time: bool) -> float:
    """Read profiler device time across old and current PyTorch APIs."""

    names = (
        ("self_device_time_total", "self_cuda_time_total")
        if self_time
        else ("device_time_total", "cuda_time_total")
    )
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_text(
    records: list[dict[str, Any]],
    status_lines: list[str],
    mechanism_rows: list[dict[str, Any]],
    op_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# KernelBench Loss Profiler Summary",
        "",
        "This diagnostic is intentionally separate from CUDA-event benchmark results. It does not create benchmark speedups, repeatability labels, or generated candidates. Historical adapter outputs are blocked from execution by default and are never paper evidence.",
        "",
        *[f"- {line}" for line in status_lines],
        "",
        "## Candidate Artifact Availability",
        "",
        "| Task | Candidate source | Metadata | Historical | Contract metadata | Current policy |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record['task']} | `{record['candidate_path']}` ({'present' if record['candidate_exists'] else 'missing'}) | `{record['result_path']}` | {record['historical_run']} | {record['candidate_contract']} / {record['recorded_policy_version']} | {record['current_policy_reason']} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism Summary",
            "",
            "| Task | Existing speedup | Candidate pattern | Attribution confidence | Caveat |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in mechanism_rows:
        lines.append(
            f"| {row['task']} | {row['existing_speedup_vs_eager']} vs eager, {row['existing_speedup_vs_compile']} vs compile | {row['candidate_pattern']} | {row['attribution_confidence']} | {row['caveat']} |"
        )
    profiled = [row for row in op_rows if row.get("status") == "profiled"]
    lines.extend(
        [
            "",
            "## Profiler Rows",
            "",
            f"- Profiled operator rows: {len(profiled)}",
            f"- Output CSV: `{OPS_CSV.relative_to(ROOT)}`",
            f"- Memory CSV: `{MEMORY_CSV.relative_to(ROOT)}`",
            f"- Mechanism CSV: `{MECHANISM_CSV.relative_to(ROOT)}`",
            "",
            "Only corrected-run profiler rows may support mechanism discussion. Historical debug rows remain excluded because candidate selection and baseline timing used the affected adapter.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
