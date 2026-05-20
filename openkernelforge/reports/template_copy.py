"""Reports for strict template-copy/adapt runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from openkernelforge.reports.run_data import load_run_bundle


def write_template_copy_report(run_dir: str | Path) -> Path:
    """Write ``template_copy_report.md`` for a template-copy run."""

    bundle = load_run_bundle(run_dir)
    path = Path(run_dir) / "template_copy_report.md"
    path.write_text(format_template_copy_report(bundle), encoding="utf-8")
    return path


def format_template_copy_report(bundle: dict[str, Any]) -> str:
    run_dir = bundle["run_dir"]
    records = [
        record
        for record in bundle["candidate_records"]
        if record.get("generation_stage") == "template_copy"
    ]
    by_task = _by_task(records)
    scores = [
        float(record.get("preserved_template_structure_score"))
        for record in records
        if record.get("preserved_template_structure_score") is not None
    ]
    lines = [
        "# OpenKernelForge Template-Copy Report",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Template-copy candidates: {len(records)}",
        f"- Median preservation score: {_fmt(median(scores) if scores else None)}",
        f"- Forbidden torch op candidates: {sum(1 for record in records if record.get('extra_torch_ops_detected'))}",
        f"- Fallback candidates: {sum(1 for record in records if record.get('fallback_detected'))}",
        "",
        "## Per-Task Best Copied Candidate",
        "",
        "| Task | Best copied candidate | Speedup vs eager | Source template speedup | Delta vs template | Preservation score | Matched template | Beat template | Source |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for task_id, task_records in sorted(by_task.items()):
        best = _best_record(task_records)
        benchmark = best.get("benchmark_summary") or {}
        delta = best.get("delta_vs_source_template")
        lines.append(
            "| {task} | {candidate} | {speedup} | {template_speedup} | {delta} | {score} | {matched} | {beat} | `{source}` |".format(
                task=task_id,
                candidate=best.get("candidate_id", "n/a"),
                speedup=_fmt(benchmark.get("speedup_vs_eager")),
                template_speedup=_fmt(best.get("source_template_speedup_vs_eager")),
                delta=_fmt(delta),
                score=_fmt(best.get("preserved_template_structure_score")),
                matched="yes" if delta is not None and abs(float(delta)) <= 0.03 else "no",
                beat="yes" if delta is not None and float(delta) > 0 else "no",
                source=best.get("candidate_path", "n/a"),
            )
        )

    violation_counts = Counter()
    for record in records:
        preservation = record.get("template_preservation") or {}
        for warning in preservation.get("warnings") or []:
            violation_counts[str(warning)] += 1
    lines.extend(["", "## Common Structure Violations", ""])
    if violation_counts:
        for warning, count in violation_counts.most_common(15):
            lines.append(f"- {warning}: {count}")
    else:
        lines.append("No preservation warnings were recorded.")
    lines.extend(["", "## Benchmark Deltas Vs Source Template", ""])
    lines.extend(
        [
            "| Task | Candidate | Requested BLOCK_SIZE | Requested num_warps | Requested contiguous policy | Delta vs template |",
            "| --- | --- | ---: | ---: | --- | ---: |",
        ]
    )
    for record in sorted(records, key=lambda item: (str(item.get("task_id")), str(item.get("candidate_id")))):
        lines.append(
            "| {task} | {candidate} | {block} | {warps} | {policy} | {delta} |".format(
                task=record.get("task_id", "unknown"),
                candidate=record.get("candidate_id", "unknown"),
                block=record.get("requested_block_size"),
                warps=record.get("requested_num_warps"),
                policy=record.get("requested_contiguous_policy"),
                delta=_fmt(record.get("delta_vs_source_template")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _by_task(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("task_id"))].append(record)
    return grouped


def _best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(records, key=_quality_key) if records else {}


def _quality_key(record: dict[str, Any]) -> tuple[int, float]:
    speedup = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
    return (1 if record.get("verification_passed") else 0, float(speedup) if speedup is not None else -1.0)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
