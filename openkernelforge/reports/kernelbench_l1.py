"""Reports and checks for the KernelBench L1 pilot."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from openkernelforge.config import RunConfig, load_config, save_config
from openkernelforge.harness.benchmarker import BenchmarkResult, benchmark_task
from openkernelforge.tasks.kernelbench_l1 import (
    KernelBenchL1Error,
    load_kernelbench_l1_tasks,
    make_candidate_provider,
)
from openkernelforge.utils.env_probe import format_environment_summary, probe_environment


def run_kernelbench_l1_check(config_path: str | Path, kernelbench_dir: str | Path) -> Path:
    config = load_config(config_path)
    raw_config = _load_raw_yaml(config_path)
    kernelbench_cfg = raw_config.get("kernelbench", {}) if isinstance(raw_config, dict) else {}
    max_tasks = int(kernelbench_cfg.get("max_tasks", 5))
    selected_task_ids = list(kernelbench_cfg.get("task_ids") or [])
    provider = make_candidate_provider(kernelbench_cfg)
    if provider.mode == "llm_later":
        provider.candidate_for_task("__probe__")

    run_dir = _new_run_dir(config.output_dir)
    save_config(config, run_dir / "config.yaml")
    env = probe_environment()
    (run_dir / "environment_probe.json").write_text(
        json.dumps(env.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        _check_execution_requirements(config, env)
        tasks = load_kernelbench_l1_tasks(
            kernelbench_dir,
            task_ids=selected_task_ids or None,
            max_tasks=max_tasks,
        )
    except Exception as exc:
        report_path = run_dir / "kernelbench_l1_check.md"
        data = {
            "run_dir": str(run_dir),
            "config": str(config_path),
            "kernelbench_dir": str(kernelbench_dir),
            "status": "failed",
            "error": str(exc),
            "environment": env.to_dict(),
            "records": [],
        }
        (run_dir / "kernelbench_l1_check.json").write_text(json.dumps(data, indent=2) + "\n")
        write_kernelbench_l1_report(run_dir, data=data)
        return report_path

    device = _select_device(config)
    for task in tasks:
        record: dict[str, Any] = {
            "task_id": task.task_id,
            "task_name": task.name,
            "op_family": task.metadata.get("op_family"),
            "shape": list(task.benchmark_shapes[0]),
            "source_path": task.metadata.get("source_path"),
            "candidate_provider": provider.mode,
            "candidate_path": None,
            "reference_ok": False,
            "benchmark_summary": None,
            "error": None,
        }
        try:
            provider_path = provider.candidate_for_task(task.task_id)
            record["candidate_path"] = str(provider_path) if provider_path else None
            inputs = task.generate_inputs(
                0,
                task.benchmark_shapes[0],
                torch.float32,
                device,
            )
            with torch.no_grad():
                output = task.reference_fn(*inputs)
            record["reference_ok"] = True
            record["output_summary"] = _output_summary(output)
            if config.benchmark.enabled:
                benchmark = benchmark_task(
                    task,
                    task.reference_fn,
                    candidate_name="reference_baseline",
                    shape=task.benchmark_shapes[0],
                    dtype=config.benchmark.dtype,
                    device=device,
                    warmup=config.benchmark.warmup,
                    repeats=config.benchmark.repeats,
                    timing_mode=config.benchmark.timing_mode,
                    independent_sessions=config.benchmark.independent_sessions,
                    cache_flush_config=config.benchmark.cache_flush,
                    bootstrap_ci_config=config.benchmark.bootstrap_ci,
                    separate_compile_time=config.benchmark.separate_compile_time,
                    stable_session_threshold=config.benchmark.stable_session_threshold,
                    enable_torch_compile=config.benchmark.enable_torch_compile,
                    torch_compile_mode=config.benchmark.torch_compile_mode,
                )
                benchmark_dict = _benchmark_to_dict(benchmark)
                record["benchmark_summary"] = _benchmark_summary(benchmark_dict)
        except Exception as exc:
            record["error"] = str(exc)
            failures.append(f"{task.task_id}: {exc}")
        records.append(record)

    data = {
        "run_dir": str(run_dir),
        "config": str(config_path),
        "kernelbench_dir": str(kernelbench_dir),
        "status": "completed" if not failures else "completed_with_failures",
        "environment": env.to_dict(),
        "tasks_loaded": len(tasks),
        "records": records,
        "failures": failures,
        "timing": {
            "timing_mode": config.benchmark.timing_mode,
            "warmup": config.benchmark.warmup,
            "repeat": config.benchmark.repeats,
            "independent_sessions": config.benchmark.independent_sessions,
            "cache_flush_enabled": config.benchmark.cache_flush.enabled,
            "bootstrap_ci_enabled": config.benchmark.bootstrap_ci.enabled,
            "torch_compile_enabled": config.benchmark.enable_torch_compile,
            "torch_compile_mode": config.benchmark.torch_compile_mode,
        },
    }
    (run_dir / "kernelbench_l1_check.json").write_text(json.dumps(data, indent=2) + "\n")
    return write_kernelbench_l1_report(run_dir, data=data)


def write_kernelbench_l1_report(
    run_dir: str | Path,
    *,
    data: dict[str, Any] | None = None,
) -> Path:
    run_path = Path(run_dir)
    if data is None:
        json_path = run_path / "kernelbench_l1_check.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))
    report_path = run_path / "kernelbench_l1_pilot_report.md"
    check_path = run_path / "kernelbench_l1_check.md"
    records = data.get("records") or []
    lines = [
        "# KernelBench L1 Pilot Report",
        "",
        "This report validates KernelBench L1 task loading and baseline timing. It contains no paid API calls, no training, and no SOTA claim.",
        "",
        "## Summary",
        "",
        f"- Status: `{data.get('status', 'unknown')}`",
        f"- Run dir: `{data.get('run_dir', run_path)}`",
        f"- KernelBench dir: `{data.get('kernelbench_dir', 'n/a')}`",
        f"- Tasks loaded: {data.get('tasks_loaded', len(records))}",
        f"- Candidate results present: {sum(1 for record in records if record.get('candidate_path'))}",
    ]
    timing = data.get("timing") or {}
    if timing:
        lines.extend(
            [
                f"- Timing mode: `{timing.get('timing_mode')}`",
                f"- Cache flush enabled: {timing.get('cache_flush_enabled')}",
                f"- Independent sessions: {timing.get('independent_sessions')}",
                f"- Repeat: {timing.get('repeat')}",
                f"- Bootstrap CI enabled: {timing.get('bootstrap_ci_enabled')}",
                f"- torch.compile enabled: {timing.get('torch_compile_enabled')}",
            ]
        )
    if data.get("error"):
        lines.extend(["", "## Error", "", str(data["error"])])

    lines.extend(["", "## Environment", "", "```text"])
    env = data.get("environment")
    if env:
        lines.append(format_environment_summary(env))
    else:
        lines.append("not recorded")
    lines.extend(["```", "", "## Tasks", ""])
    lines.append("| Task | Op family | Shape | Eager median ms | torch.compile median ms | Ready | Failure |")
    lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
    for record in records:
        summary = record.get("benchmark_summary") or {}
        ready = "yes" if record.get("reference_ok") and not record.get("error") else "no"
        lines.append(
            "| `{task}` | {family} | `{shape}` | {eager} | {compile} | {ready} | {failure} |".format(
                task=record.get("task_id"),
                family=record.get("op_family") or "",
                shape=record.get("shape"),
                eager=_fmt(summary.get("eager_median_ms")),
                compile=_fmt(summary.get("torch_compile_median_ms")),
                ready=ready,
                failure=(record.get("error") or "").replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## Candidate Results",
            "",
            "Candidate generation is intentionally optional in this sprint. `candidate_provider=none` validates task loading, eager references, and optional `torch.compile` baselines only.",
            "",
            "- Single-run wins: none recorded by baseline-only check.",
            "- Repeat-stable wins: none recorded by baseline-only check.",
            "- Single-run-only wins: none recorded by baseline-only check.",
            "- Unstable fraction: not applicable without candidate results.",
            "- Failure taxonomy: reference/load/benchmark failures are reported above.",
            "",
            "## Readiness",
            "",
            "Ready for candidate generation if all selected tasks show `Ready=yes` and baseline timing summaries are present.",
        ]
    )
    text = "\n".join(lines) + "\n"
    report_path.write_text(text, encoding="utf-8")
    check_path.write_text(text, encoding="utf-8")
    return check_path


def _load_raw_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _new_run_dir(output_dir: str) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = root / stamp
    counter = 1
    while path.exists():
        path = root / f"{stamp}_{counter}"
        counter += 1
    path.mkdir(parents=True)
    return path


def _check_execution_requirements(config: RunConfig, env: Any) -> None:
    if config.execution.require_cuda and not env.cuda_available:
        raise RuntimeError("Config requires CUDA, but CUDA is unavailable.")
    if config.execution.require_triton and not env.triton_available:
        raise RuntimeError("Config requires Triton, but Triton is unavailable.")
    if config.execution.require_tiny_triton_kernel and not env.tiny_triton_kernel_passed:
        raise RuntimeError("Config requires a passing tiny Triton kernel, but the probe failed.")


def _select_device(config: RunConfig) -> str:
    if config.benchmark.device != "auto":
        return config.benchmark.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _benchmark_to_dict(result: BenchmarkResult) -> dict[str, Any]:
    return _to_jsonable(dataclasses.asdict(result))


def _benchmark_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    eager = benchmark.get("eager") or {}
    candidate = benchmark.get("candidate") or {}
    compiled = benchmark.get("torch_compile") or {}
    return {
        "shape": benchmark.get("shape"),
        "dtype": benchmark.get("dtype"),
        "device": benchmark.get("device"),
        "eager_median_ms": eager.get("median_ms"),
        "torch_compile_median_ms": compiled.get("median_ms"),
        "torch_compile_mode": benchmark.get("torch_compile_mode"),
        "timing_mode": benchmark.get("timing_mode"),
        "warmup": benchmark.get("warmup"),
        "repeat": benchmark.get("repeats"),
        "independent_sessions": benchmark.get("independent_sessions"),
        "cache_flush_enabled": benchmark.get("cache_flush_enabled"),
        "cache_flush_performed": benchmark.get("cache_flush_performed"),
        "eager_ms_summary": benchmark.get("eager_ms_summary"),
        "torch_compile_ms_summary": benchmark.get("torch_compile_ms_summary"),
        "speedup_vs_compile": benchmark.get("speedup_vs_torch_compile"),
        "compile_time_ms": benchmark.get("compile_time_ms"),
        "measurement_warnings": benchmark.get("measurement_warnings") or [],
        "benchmark_error": benchmark.get("benchmark_error"),
        "compile_error": benchmark.get("compile_error"),
        "reference_self_median_ms": candidate.get("median_ms"),
    }


def _output_summary(output: Any) -> dict[str, Any]:
    if isinstance(output, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(output.shape),
            "dtype": str(output.dtype).replace("torch.", ""),
            "device": str(output.device),
        }
    if isinstance(output, (list, tuple)):
        return {"type": type(output).__name__, "length": len(output)}
    return {"type": type(output).__name__}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.device):
        return str(value)
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
