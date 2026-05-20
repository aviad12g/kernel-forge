"""Focused deterministic sweep reports."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from openkernelforge.reports.run_data import load_run_bundle, read_artifact


def write_focused_sweep_report(
    run_dir: str | Path,
    *,
    shapeaware_run: str | Path | None = None,
    template_copy_wide_run: str | Path | None = None,
) -> Path:
    bundle = load_run_bundle(run_dir)
    shapeaware_bundle = load_run_bundle(shapeaware_run) if shapeaware_run else None
    copy_bundle = load_run_bundle(template_copy_wide_run) if template_copy_wide_run else None
    path = Path(run_dir) / "focused_sweep_report.md"
    path.write_text(
        format_focused_sweep_report(bundle, shapeaware_bundle, copy_bundle),
        encoding="utf-8",
    )
    return path


def write_focused_sweep_seed_analysis(
    *,
    shapeaware_run: str | Path,
    template_copy_wide_run: str | Path,
    out_path: str | Path,
) -> Path:
    shapeaware = load_run_bundle(shapeaware_run)
    copy = load_run_bundle(template_copy_wide_run)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_seed_analysis(shapeaware, copy), encoding="utf-8")
    return path


def format_seed_analysis(shapeaware: dict[str, Any], copy: dict[str, Any]) -> str:
    shape_best = _best_by_task(shapeaware["candidate_records"])
    copy_best = _best_by_task(copy["candidate_records"])
    seed_for_task = {
        "vector_add": ("shape-aware template", shape_best.get("vector_add")),
        "bias_relu": ("shape-aware template", shape_best.get("bias_relu")),
        "relu": ("template-copy-wide", copy_best.get("relu")),
    }
    lines = [
        "# Focused Sweep Seed Analysis",
        "",
        f"- Shape-aware run: `{shapeaware['run_dir']}`",
        f"- Template-copy-wide run: `{copy['run_dir']}`",
        "",
        "| Task | Seed source | Speedup vs eager | Candidate median ms | Candidate path | Metadata |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for task_id, (source_name, record) in seed_for_task.items():
        benchmark = (record or {}).get("benchmark_summary") or {}
        lines.append(
            "| {task} | {source} | {speedup} | {median} | `{path}` | {metadata} |".format(
                task=task_id,
                source=source_name,
                speedup=_fmt(benchmark.get("speedup_vs_eager")),
                median=_fmt(benchmark.get("candidate_median_ms")),
                path=(record or {}).get("candidate_path", "n/a"),
                metadata=_metadata_text(record or {}),
            )
        )
    lines.extend(["", "## Source Summaries", ""])
    for task_id, (_, record) in seed_for_task.items():
        if not record:
            continue
        source = read_artifact(record.get("candidate_path"), run_dir=shapeaware["run_dir"])
        if not source:
            source = read_artifact(record.get("candidate_path"), run_dir=copy["run_dir"])
        lines.extend(
            [
                f"### {task_id}",
                "",
                f"- Seed reason: highest observed speed for `{task_id}` among the selected prior runs.",
                f"- Source path: `{record.get('candidate_path')}`",
                "",
                "```python",
                "\n".join(source.splitlines()[:35]),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def format_focused_sweep_report(
    bundle: dict[str, Any],
    shapeaware_bundle: dict[str, Any] | None = None,
    copy_bundle: dict[str, Any] | None = None,
) -> str:
    records = bundle["candidate_records"]
    by_task = _by_task(records)
    shape_best = _best_by_task(shapeaware_bundle["candidate_records"]) if shapeaware_bundle else {}
    copy_best = _best_by_task(copy_bundle["candidate_records"]) if copy_bundle else {}
    lines = [
        "# OpenKernelForge Focused Sweep Report",
        "",
        f"- Run dir: `{bundle['run_dir']}`",
        f"- Candidates: {len(records)}",
        f"- Benchmarked: {sum(1 for record in records if record.get('benchmark_summary'))}",
        "",
        "## Per-Task Outcome",
        "",
        "| Task | Focused best eager | Focused best compile | Shape-aware best | Copy-wide best | >=1.0x eager | >=1.0x compile | Recommendation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for task_id, task_records in sorted(by_task.items()):
        best = _best_record(task_records, "speedup_vs_eager")
        best_compile = _best_record(task_records, "speedup_vs_torch_compile")
        speedups = _speedups(task_records, "speedup_vs_eager")
        compile_speedups = _speedups(task_records, "speedup_vs_torch_compile")
        shape_speed = ((shape_best.get(task_id) or {}).get("benchmark_summary") or {}).get("speedup_vs_eager")
        copy_speed = ((copy_best.get(task_id) or {}).get("benchmark_summary") or {}).get("speedup_vs_eager")
        lines.append(
            "| {task} | {eager} | {compile} | {shape} | {copy} | {fast} | {fast_compile} | {recommendation} |".format(
                task=task_id,
                eager=_fmt(((best or {}).get("benchmark_summary") or {}).get("speedup_vs_eager")),
                compile=_fmt(((best_compile or {}).get("benchmark_summary") or {}).get("speedup_vs_torch_compile")),
                shape=_fmt(shape_speed),
                copy=_fmt(copy_speed),
                fast=sum(1 for value in speedups if value >= 1.0),
                fast_compile=sum(1 for value in compile_speedups if value >= 1.0),
                recommendation=_recommendation(task_records, shape_speed, copy_speed),
            )
        )

    lines.extend(["", "## Per-Task Top 20", ""])
    for task_id, task_records in sorted(by_task.items()):
        lines.extend(
            [
                f"### {task_id}",
                "",
                "| Rank | Candidate | Speedup eager | Speedup compile | BLOCK_SIZE | warps | stages | n_mode | feature_mode | allocation | Source |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        for rank, record in enumerate(sorted(task_records, key=_eager_key, reverse=True)[:20], start=1):
            benchmark = record.get("benchmark_summary") or {}
            lines.append(
                "| {rank} | {candidate} | {eager} | {compile} | {block} | {warps} | {stages} | {n_mode} | {feature} | {alloc} | `{source}` |".format(
                    rank=rank,
                    candidate=record.get("candidate_id"),
                    eager=_fmt(benchmark.get("speedup_vs_eager")),
                    compile=_fmt(benchmark.get("speedup_vs_torch_compile")),
                    block=record.get("block_size"),
                    warps=record.get("num_warps"),
                    stages=record.get("num_stages"),
                    n_mode=record.get("n_elements_mode"),
                    feature=record.get("feature_dim_mode"),
                    alloc=record.get("output_allocation_policy"),
                    source=record.get("candidate_path"),
                )
            )
        lines.append("")

    lines.extend(["## Sensitivity Tables", ""])
    for field in (
        "block_size",
        "num_warps",
        "num_stages",
        "n_elements_mode",
        "feature_dim_mode",
        "contiguous_policy",
        "output_allocation_policy",
    ):
        lines.extend([f"### {field}", "", "| Value | Count | Best eager | Median eager | Best compile |", "| --- | ---: | ---: | ---: | ---: |"])
        for value, field_records in sorted(_group_by(records, field).items(), key=lambda item: str(item[0])):
            eager = _speedups(field_records, "speedup_vs_eager")
            compile_values = _speedups(field_records, "speedup_vs_torch_compile")
            lines.append(
                f"| {value} | {len(field_records)} | {_fmt(max(eager) if eager else None)} | "
                f"{_fmt(median(eager) if eager else None)} | {_fmt(max(compile_values) if compile_values else None)} |"
            )
        lines.append("")
    return "\n".join(lines)


def _by_task(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("task_id"))].append(record)
    return grouped


def _best_by_task(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        speedup = ((record.get("benchmark_summary") or {}).get("speedup_vs_eager"))
        if speedup is None:
            continue
        task_id = str(record.get("task_id"))
        if task_id not in best or float(speedup) > float((best[task_id].get("benchmark_summary") or {}).get("speedup_vs_eager")):
            best[task_id] = record
    return best


def _best_record(records: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    scored = [
        (float(value), record)
        for record in records
        if (value := (record.get("benchmark_summary") or {}).get(metric)) is not None
    ]
    return max(scored, key=lambda item: item[0])[1] if scored else None


def _speedups(records: list[dict[str, Any]], metric: str) -> list[float]:
    return [
        float(value)
        for record in records
        if (value := (record.get("benchmark_summary") or {}).get(metric)) is not None
    ]


def _eager_key(record: dict[str, Any]) -> float:
    value = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
    return float(value) if value is not None else -1.0


def _group_by(records: list[dict[str, Any]], field: str) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record.get(field)].append(record)
    return grouped


def _recommendation(records: list[dict[str, Any]], shape_speed: Any, copy_speed: Any) -> str:
    best = _best_record(records, "speedup_vs_eager")
    focused = ((best or {}).get("benchmark_summary") or {}).get("speedup_vs_eager")
    if focused is not None and float(focused) >= 1.0:
        return "include in dataset and add profiler validation"
    prior = max(float(value) for value in (shape_speed, copy_speed) if value is not None) if any(value is not None for value in (shape_speed, copy_speed)) else None
    if focused is not None and prior is not None and float(focused) > prior:
        return "continue optimizing this task"
    return "add profiler or move to fused tasks"


def _metadata_text(record: dict[str, Any]) -> str:
    keys = [
        "block_size",
        "num_warps",
        "num_stages",
        "contiguous_policy",
        "output_allocation_policy",
        "shape_specialized",
        "feature_dim_mode",
        "n_elements_mode",
    ]
    return ", ".join(f"{key}={record.get(key)}" for key in keys)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
