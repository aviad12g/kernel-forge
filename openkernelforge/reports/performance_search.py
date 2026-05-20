"""Performance-search reporting for correct-but-slow kernel candidates."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from openkernelforge.reports.failure_taxonomy import classify_candidate_record
from openkernelforge.reports.run_data import load_run_bundle


def write_performance_search_report(run_dir: str | Path) -> Path:
    """Write ``performance_search_report.md`` for a run."""

    bundle = load_run_bundle(run_dir)
    path = Path(run_dir) / "performance_search_report.md"
    path.write_text(format_performance_search_report(bundle), encoding="utf-8")
    return path


def format_performance_search_report(bundle: dict[str, Any]) -> str:
    run_dir = bundle["run_dir"]
    candidates = bundle["candidate_records"]
    config = bundle.get("config") or {}
    agent = config.get("agent") or {}
    perf = agent.get("performance_search") or {}
    by_task = _by_task(candidates)

    lines = [
        "# OpenKernelForge Performance Search Report",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Enabled: {'yes' if perf.get('enabled') else 'no'}",
        f"- Performance prompt version: `{agent.get('performance_prompt_version', 'n/a')}`",
        f"- Target speedup vs eager: {perf.get('target_speedup_vs_eager', 'n/a')}",
        f"- Max rounds: {perf.get('max_rounds', 'n/a')}",
        f"- Candidates per round: {perf.get('candidates_per_round', 'n/a')}",
        "",
        "## Per-Task Search Outcome",
        "",
        "| Task | Best initial | Best final | Delta | Target reached | Search candidates | Verified | Benchmarked | Common failures |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for task_id, records in sorted(by_task.items()):
        initial = [record for record in records if record.get("generation_stage", "initial") != "performance_search"]
        search = [record for record in records if record.get("generation_stage") == "performance_search"]
        best_initial = _best_speedup(initial)
        best_final = _best_speedup(records)
        delta = best_final - best_initial if best_initial is not None and best_final is not None else None
        failures = Counter(
            classify_candidate_record(record).failure_type
            for record in search
            if not record.get("verification_passed")
        )
        lines.append(
            "| {task} | {initial} | {final} | {delta} | {target} | {search_count} | {verified} | {benchmarked} | {failures} |".format(
                task=task_id,
                initial=_fmt(best_initial),
                final=_fmt(best_final),
                delta=_fmt(delta),
                target="yes" if any(record.get("target_reached") for record in records) else "no",
                search_count=len(search),
                verified=sum(1 for record in search if record.get("verification_passed")),
                benchmarked=sum(1 for record in search if record.get("benchmark_summary")),
                failures=", ".join(f"{k}:{v}" for k, v in failures.most_common()) or "none",
            )
        )

    search_records = [record for record in candidates if record.get("generation_stage") == "performance_search"]
    speedups = [
        float((record.get("benchmark_summary") or {}).get("speedup_vs_eager"))
        for record in search_records
        if (record.get("benchmark_summary") or {}).get("speedup_vs_eager") is not None
    ]
    lines.extend(
        [
            "",
            "## Aggregate Search Counts",
            "",
            f"- Generated optimization candidates: {len(search_records)}",
            f"- Verified optimization candidates: {sum(1 for record in search_records if record.get('verification_passed'))}",
            f"- Benchmarked optimization candidates: {sum(1 for record in search_records if record.get('benchmark_summary'))}",
            f"- Improved over parent: {sum(1 for record in search_records if record.get('improved_over_parent'))}",
            f"- Target reached tasks: {sum(1 for records in by_task.values() if any(record.get('target_reached') for record in records))}",
            f"- Median search speedup vs eager: {_fmt(median(speedups) if speedups else None)}",
            "",
            "## Useful Optimization Pairs",
            "",
        ]
    )
    pairs = [
        record
        for record in search_records
        if record.get("verification_passed") and record.get("improved_over_parent")
    ]
    if pairs:
        for record in pairs[:20]:
            benchmark = record.get("benchmark_summary") or {}
            lines.append(
                f"- {record.get('task_id')} {record.get('parent_candidate_id')} -> "
                f"{record.get('candidate_id')}: parent_speedup={_fmt(record.get('parent_speedup_vs_eager'))}, "
                f"child_speedup={_fmt(benchmark.get('speedup_vs_eager'))}"
            )
    else:
        lines.append("No improved parent-child optimization pairs were recorded.")
    lines.append("")
    return "\n".join(lines)


def _by_task(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        grouped[str(record.get("task_id"))].append(record)
    return grouped


def _best_speedup(records: list[dict[str, Any]]) -> float | None:
    values = [
        float((record.get("benchmark_summary") or {}).get("speedup_vs_eager"))
        for record in records
        if (record.get("benchmark_summary") or {}).get("speedup_vs_eager") is not None
    ]
    return max(values) if values else None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
