"""Repeatability checks for top benchmark candidates."""

from __future__ import annotations

import json
import statistics
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from openkernelforge.config import RunConfig
from openkernelforge.harness.benchmarker import benchmark_task
from openkernelforge.harness.policy import check_candidate_policy
from openkernelforge.harness.sandbox import load_candidate_from_path, unload_candidate
from openkernelforge.reports.run_data import load_run_bundle
from openkernelforge.tasks.simple_tasks import get_task


def write_repeatability_report(run_dir: str | Path, *, top_k: int = 5, repeats: int = 5) -> tuple[Path, Path]:
    """Rebenchmark top candidates per task and write repeatability artifacts."""

    run_path = Path(run_dir)
    bundle = load_run_bundle(run_path)
    config = RunConfig.from_dict(bundle.get("config") or {})
    rows = collect_repeatability_results(bundle, config=config, top_k=top_k, repeats=repeats)
    json_path = run_path / "repeatability_results.json"
    json_path.write_text(json.dumps({"top_k": top_k, "repeats": repeats, "results": rows}, indent=2) + "\n", encoding="utf-8")
    report_path = run_path / "repeatability_report.md"
    report_path.write_text(format_repeatability_report(run_path, rows), encoding="utf-8")
    return report_path, json_path


def collect_repeatability_results(
    bundle: dict[str, Any],
    *,
    config: RunConfig,
    top_k: int,
    repeats: int,
) -> list[dict[str, Any]]:
    records = _top_candidates_by_task(bundle["candidate_records"], top_k=top_k)
    rows: list[dict[str, Any]] = []
    for record in records:
        task = get_task(str(record.get("task_id")))
        candidate_path = _resolve_path(record.get("candidate_path"), bundle["run_dir"])
        speedups: list[float] = []
        compile_speedups: list[float] = []
        candidate_medians: list[float] = []
        errors: list[str] = []
        for _ in range(repeats):
            loaded = None
            try:
                source = candidate_path.read_text(encoding="utf-8", errors="strict")
                policy = check_candidate_policy(
                    source,
                    allow_torch_fallback=config.agent.allow_torch_fallback,
                    require_triton=not config.agent.allow_torch_fallback,
                )
                if not policy.passed:
                    raise RuntimeError(
                        "candidate failed current policy before repeatability timing: "
                        f"{policy.rejection_reason}"
                    )
                loaded = load_candidate_from_path(candidate_path)
                if loaded.forward is None:
                    raise RuntimeError(f"candidate has no module-level forward: {candidate_path}")
                summary = record.get("benchmark_summary") or {}
                shape = tuple(summary.get("shape") or task.benchmark_shapes[0])
                benchmark = benchmark_task(
                    task,
                    loaded.forward,
                    candidate_name=str(record.get("candidate_name") or record.get("candidate_id")),
                    shape=shape,
                    dtype=config.benchmark.dtype,
                    device=config.benchmark.device,
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
                if benchmark.benchmark_error:
                    errors.append(benchmark.benchmark_error)
                    continue
                if benchmark.speedup_vs_eager is not None:
                    speedups.append(float(benchmark.speedup_vs_eager))
                if benchmark.speedup_vs_torch_compile is not None:
                    compile_speedups.append(float(benchmark.speedup_vs_torch_compile))
                aggregate_candidate = benchmark.candidate_ms_summary or {}
                if aggregate_candidate.get("median_ms") is not None:
                    candidate_medians.append(float(aggregate_candidate["median_ms"]))
                if benchmark.compile_error:
                    errors.append("torch.compile error:\n" + benchmark.compile_error)
            except Exception:
                errors.append(traceback.format_exc())
            finally:
                if loaded is not None:
                    unload_candidate(loaded)
        stats = _stats(speedups)
        stable = _is_stable(speedups)
        rows.append(
            {
                "task_id": record.get("task_id"),
                "candidate_id": record.get("candidate_id"),
                "candidate_path": record.get("candidate_path"),
                "original_speedup_vs_eager": (record.get("benchmark_summary") or {}).get("speedup_vs_eager"),
                "original_speedup_vs_torch_compile": (record.get("benchmark_summary") or {}).get("speedup_vs_torch_compile"),
                "template_metadata": {
                    key: record.get(key)
                    for key in (
                        "block_size",
                        "num_warps",
                        "num_stages",
                        "contiguous_policy",
                        "output_allocation_policy",
                        "n_elements_mode",
                        "feature_dim_mode",
                        "shape_specialized",
                    )
                },
                "speedup_values": speedups,
                "speedup_vs_compile_values": compile_speedups,
                "candidate_median_ms_values": candidate_medians,
                "stats": stats,
                "compile_stats": _stats(compile_speedups),
                "stable": stable,
                "label": classify_repeatability_label(
                    original_speedup=(record.get("benchmark_summary") or {}).get("speedup_vs_eager"),
                    stats=stats,
                    stable=stable,
                ),
                "errors": errors[:3],
            }
        )
    return rows


def format_repeatability_report(run_dir: str | Path, rows: list[dict[str, Any]]) -> str:
    lines = [
        "# OpenKernelForge Repeatability Report",
        "",
        f"- Run dir: `{run_dir}`",
        "- Stability threshold: coefficient of variation <= 0.10 over speedup vs eager",
        "- Labels: REPEAT_STABLE_WIN, SINGLE_RUN_ONLY_WIN, UNSTABLE, BELOW_EAGER, INSUFFICIENT_DATA",
        "",
        "| Task | Candidate | Label | Original speedup | Median repeat speedup | Mean | Std | Min | Max | CV | Stable | Path |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        stats = row.get("stats") or {}
        lines.append(
            "| {task} | {candidate} | {label} | {original} | {median} | {mean} | {std} | {minv} | {maxv} | {cv} | {stable} | `{path}` |".format(
                task=row.get("task_id"),
                candidate=row.get("candidate_id"),
                label=row.get("label") or "INSUFFICIENT_DATA",
                original=_fmt(row.get("original_speedup_vs_eager")),
                median=_fmt(stats.get("median")),
                mean=_fmt(stats.get("mean")),
                std=_fmt(stats.get("std")),
                minv=_fmt(stats.get("min")),
                maxv=_fmt(stats.get("max")),
                cv=_fmt(stats.get("coefficient_of_variation")),
                stable="yes" if row.get("stable") else "no",
                path=row.get("candidate_path"),
            )
        )
    lines.extend(["", "## Best Repeatability By Task", ""])
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task_id"))].append(row)
    for task_id, task_rows in sorted(by_task.items()):
        valid = [row for row in task_rows if (row.get("stats") or {}).get("median") is not None]
        if not valid:
            lines.append(f"- {task_id}: no successful repeatability measurements")
            continue
        best = max(valid, key=_repeat_median)
        lines.append(
            f"- {task_id}: `{best.get('candidate_id')}` median repeat "
            f"{_fmt((best.get('stats') or {}).get('median'))}x, label={best.get('label')}, "
            f"stable={'yes' if best.get('stable') else 'no'}"
        )
    lines.append("")
    return "\n".join(lines)


def _top_candidates_by_task(records: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        benchmark = record.get("benchmark_summary") or {}
        if not (record.get("policy_passed") and record.get("verification_passed")):
            continue
        if benchmark.get("speedup_vs_eager") is None:
            continue
        by_task[str(record.get("task_id"))].append(record)
    selected: list[dict[str, Any]] = []
    for records_for_task in by_task.values():
        selected.extend(
            sorted(
                records_for_task,
                key=_record_speedup_vs_eager,
                reverse=True,
            )[:top_k]
        )
    return selected


def _repeat_median(row: dict[str, Any]) -> float:
    value = (row.get("stats") or {}).get("median")
    if value is None:
        raise ValueError("Repeatability row has no median")
    return float(value)


def _record_speedup_vs_eager(record: dict[str, Any]) -> float:
    value = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
    if value is None:
        raise ValueError("Candidate record has no eager speedup")
    return float(value)


def classify_repeatability_label(
    *,
    original_speedup: Any,
    stats: dict[str, Any],
    stable: bool,
) -> str:
    """Classify whether a speedup survived repeatability measurement."""

    median = stats.get("median")
    if median is None:
        return "INSUFFICIENT_DATA"
    median_value = float(median)
    original_value = float(original_speedup) if original_speedup is not None else None
    if median_value >= 1.0 and stable:
        return "REPEAT_STABLE_WIN"
    if original_value is not None and original_value >= 1.0 and median_value < 1.0:
        return "SINGLE_RUN_ONLY_WIN"
    if median_value >= 1.0:
        return "UNSTABLE"
    return "BELOW_EAGER"


def _resolve_path(path_value: Any, run_dir: str | Path) -> Path:
    path = Path(str(path_value))
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = Path(run_dir) / path
        if candidate.exists():
            return candidate
    return path


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "coefficient_of_variation": None,
        }
    mean = float(statistics.fmean(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "median": float(statistics.median(values)),
        "std": std,
        "min": float(min(values)),
        "max": float(max(values)),
        "coefficient_of_variation": abs(std / mean) if mean else None,
    }


def _is_stable(values: list[float]) -> bool:
    stats = _stats(values)
    cv = stats.get("coefficient_of_variation")
    return cv is not None and float(cv) <= 0.10


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
