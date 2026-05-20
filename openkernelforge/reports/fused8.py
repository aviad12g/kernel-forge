"""Reports for the internal fused8 benchmark."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from openkernelforge.reports.failure_taxonomy import classify_candidate_record
from openkernelforge.reports.run_data import load_run_bundle
from openkernelforge.tasks.simple_tasks import get_task


FUSED8_TASKS = [
    "bias_relu",
    "sigmoid_mul",
    "add_relu",
    "residual_add_relu",
    "bias_gelu",
    "row_sum",
    "layernorm_small",
    "rmsnorm_small",
]


def write_fused8_report(run_dir: str | Path) -> Path:
    bundle = load_run_bundle(run_dir)
    path = Path(run_dir) / "fused8_report.md"
    path.write_text(format_fused8_report(bundle), encoding="utf-8")
    return path


def format_fused8_report(bundle: dict[str, Any]) -> str:
    records = bundle["candidate_records"]
    by_task = _by_task(records)
    environment = bundle.get("environment") or {}
    repeatability = _load_repeatability(bundle["run_dir"])
    lines = [
        "# OpenKernelForge Fused8 Report",
        "",
        "This is an internal OpenKernelForge fused-task benchmark, not KernelBench and not a SOTA claim.",
        "",
        f"- Run dir: `{bundle['run_dir']}`",
        f"- Environment viability: `{environment.get('viability', 'n/a')}`",
        f"- Candidates: {len(records)}",
        f"- Verified: {sum(1 for record in records if record.get('verification_passed'))}/{len(records)}",
        f"- Benchmarked: {sum(1 for record in records if record.get('benchmark_summary'))}",
        f"- Repeatability present: {'yes' if repeatability else 'no'}",
        "",
        "## Task Shapes",
        "",
        "| Task | Shape | Family | Prompt hints |",
        "| --- | --- | --- | --- |",
    ]
    for task_id in FUSED8_TASKS:
        try:
            task = get_task(task_id)
            shape = task.benchmark_shapes[0]
            metadata = task.metadata
            hints = "; ".join(metadata.get("prompt_hints") or [])
            lines.append(f"| {task_id} | `{shape}` | {metadata.get('task_family', 'n/a')} | {hints} |")
        except KeyError:
            lines.append(f"| {task_id} | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Per-Task Best",
            "",
            "| Task | Candidates | Verified | Compile/runtime failures | Eager ms | torch.compile ms | Best candidate ms | Speedup eager | Speedup compile | Best candidate | Promising? |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for task_id in FUSED8_TASKS:
        task_records = by_task.get(task_id, [])
        best = _best_record(task_records)
        benchmark = (best or {}).get("benchmark_summary") or {}
        failures = sum(1 for record in task_records if not record.get("verification_passed"))
        speedup = benchmark.get("speedup_vs_eager")
        lines.append(
            "| {task} | {count} | {verified} | {failures} | {eager} | {compile} | {candidate_ms} | {speedup} | {speedup_compile} | `{candidate}` | {promising} |".format(
                task=task_id,
                count=len(task_records),
                verified=sum(1 for record in task_records if record.get("verification_passed")),
                failures=failures,
                eager=_fmt(benchmark.get("eager_median_ms")),
                compile=_fmt(benchmark.get("torch_compile_median_ms")),
                candidate_ms=_fmt(benchmark.get("candidate_median_ms")),
                speedup=_fmt(speedup),
                speedup_compile=_fmt(benchmark.get("speedup_vs_torch_compile")),
                candidate=(best or {}).get("candidate_path", "n/a"),
                promising=_promising_label(speedup),
            )
        )

    lines.extend(["", "## Failure Taxonomy", ""])
    counts = Counter(classify_candidate_record(record).failure_type for record in records)
    if counts:
        for label, count in sorted(counts.items()):
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Template Metadata For Best Candidates", ""])
    for task_id in FUSED8_TASKS:
        best = _best_record(by_task.get(task_id, []))
        if not best:
            continue
        metadata = {
            key: best.get(key)
            for key in (
                "template_family",
                "block_size",
                "reduction_block_size",
                "num_warps",
                "num_stages",
                "contiguous_policy",
                "output_allocation_policy",
                "n_elements_mode",
                "feature_dim_mode",
                "reduction_axis",
            )
        }
        lines.append(f"- {task_id}: `{json.dumps(metadata, sort_keys=True)}`")

    lines.extend(["", "## Recommendation", ""])
    for task_id in FUSED8_TASKS:
        best = _best_record(by_task.get(task_id, []))
        speedup = ((best or {}).get("benchmark_summary") or {}).get("speedup_vs_eager")
        if speedup is not None and float(speedup) >= 1.0:
            lines.append(f"- {task_id}: promising Triton target; run repeatability and compare with LLM/template-guided variants.")
        elif speedup is not None and float(speedup) >= 0.8:
            lines.append(f"- {task_id}: near-promising; keep in template/LLM search.")
        else:
            lines.append(f"- {task_id}: not yet promising in this run.")
    lines.append("")
    return "\n".join(lines)


def _by_task(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("task_id"))].append(record)
    return grouped


def _best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [
        (float(value), record)
        for record in records
        if (value := (record.get("benchmark_summary") or {}).get("speedup_vs_eager")) is not None
    ]
    return max(scored, key=lambda item: item[0])[1] if scored else None


def _load_repeatability(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "repeatability_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _promising_label(speedup: Any) -> str:
    if speedup is None:
        return "unknown"
    value = float(speedup)
    if value >= 1.0:
        return "beats eager"
    if value >= 0.8:
        return "near eager"
    return "slow"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
