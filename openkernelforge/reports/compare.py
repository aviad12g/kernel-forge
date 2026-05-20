"""Compare multiple OpenKernelForge run directories."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from openkernelforge.reports.summarize import load_results


def compare_runs_markdown(run_dirs: list[str | Path]) -> str:
    """Return a Markdown table comparing run-level metrics."""

    rows = [_summarize_run(Path(run_dir)) for run_dir in run_dirs]
    lines = [
        "| Run dir | Agent/backend/model | Tasks | Candidates | Policy pass rate | Verification pass rate | Benchmarked | Selected correct tasks | Median speedup eager | Median speedup compile | Wall time s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run_dir} | {label} | {tasks} | {candidates} | {policy} | {verification} | {benchmarked} | {selected} | {speedup_eager} | {speedup_compile} | {wall} |".format(
                run_dir=row["run_dir"],
                label=row["label"],
                tasks=row["tasks"],
                candidates=row["candidates"],
                policy=_fmt_rate(row["policy_pass_rate"]),
                verification=_fmt_rate(row["verification_pass_rate"]),
                benchmarked=row["benchmarked"],
                selected=row["selected_correct_tasks"],
                speedup_eager=_fmt_float(row["median_speedup_vs_eager"]),
                speedup_compile=_fmt_float(row["median_speedup_vs_torch_compile"]),
                wall=_fmt_float(row["wall_time_s"]),
            )
        )
    return "\n".join(lines) + "\n"


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    records = load_results(run_dir)
    task_records = [
        record for record in records if record.get("record_type", "task_summary") != "candidate"
    ]
    candidate_records = [record for record in records if record.get("record_type") == "candidate"]
    if not candidate_records:
        candidate_records = [
            candidate
            for record in task_records
            for candidate in record.get("candidate_records", [])
        ]

    metadata = _load_metadata(run_dir)
    first_candidate = candidate_records[0] if candidate_records else {}
    first_task = task_records[0] if task_records else {}
    agent = first_candidate.get("agent_type") or first_task.get("agent_type") or "unknown"
    backend = first_candidate.get("backend") or first_task.get("backend") or "unknown"
    model = first_candidate.get("model") or "n/a"

    policy_passed = sum(1 for record in candidate_records if record.get("policy_passed"))
    verification_passed = sum(1 for record in candidate_records if record.get("verification_passed"))
    benchmarked = sum(1 for record in candidate_records if record.get("benchmark_summary"))
    selected_correct = sum(
        1 for record in task_records if record.get("verification", {}).get("passed")
    )
    speedups_eager = [
        (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
        for record in candidate_records
    ]
    speedups_compile = [
        (record.get("benchmark_summary") or {}).get("speedup_vs_torch_compile")
        for record in candidate_records
    ]

    return {
        "run_dir": str(run_dir),
        "label": f"{agent}/{backend}/{model}",
        "tasks": len(task_records),
        "candidates": len(candidate_records),
        "policy_pass_rate": _rate(policy_passed, len(candidate_records)),
        "verification_pass_rate": _rate(verification_passed, len(candidate_records)),
        "benchmarked": benchmarked,
        "selected_correct_tasks": selected_correct,
        "median_speedup_vs_eager": _median_optional(speedups_eager),
        "median_speedup_vs_torch_compile": _median_optional(speedups_compile),
        "wall_time_s": metadata.get("duration_s"),
    }


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _median_optional(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(statistics.median(numeric))


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
