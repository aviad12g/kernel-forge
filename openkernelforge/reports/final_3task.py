"""Final conclusion report for the first three simple GPU tasks."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

from openkernelforge.reports.run_data import load_run_bundle
from openkernelforge.reports.skipped_variants import load_skipped_variants


def write_final_3task_report(
    *,
    base_template: str | Path,
    shapeaware: str | Path,
    template_copy_wide: str | Path,
    focused: str | Path,
    clean_focused: str | Path,
    out: str | Path = "runs/final_3task_conclusion.md",
) -> Path:
    runs = {
        "base_template": load_run_bundle(base_template),
        "shapeaware": load_run_bundle(shapeaware),
        "template_copy_wide": load_run_bundle(template_copy_wide),
        "focused": load_run_bundle(focused),
        "clean_focused": load_run_bundle(clean_focused),
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(format_final_3task_report(runs), encoding="utf-8")
    return out_path


def format_final_3task_report(runs: dict[str, dict[str, Any]]) -> str:
    best_by_run = {name: _best_by_task(bundle["candidate_records"]) for name, bundle in runs.items()}
    tasks = sorted({task for task_best in best_by_run.values() for task in task_best})
    clean = runs["clean_focused"]
    repeatability = _load_repeatability(clean["run_dir"])
    lines = [
        "# OpenKernelForge Final 3-Task Conclusion",
        "",
        "This report is limited to three simple internal OpenKernelForge tasks. It is not a SOTA claim, not KernelBench, and not a training result.",
        "",
        "## Runs",
        "",
        "| Label | Run dir | Candidates | Verified | Benchmarked | Median speedup vs eager | Skipped variants |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, bundle in runs.items():
        candidates = bundle["candidate_records"]
        speedups = _speedups(candidates)
        skipped = len(load_skipped_variants(bundle["run_dir"]))
        lines.append(
            "| {name} | `{run}` | {candidates} | {verified} | {benchmarked} | {median} | {skipped} |".format(
                name=name,
                run=bundle["run_dir"],
                candidates=len(candidates),
                verified=sum(1 for record in candidates if record.get("verification_passed")),
                benchmarked=sum(1 for record in candidates if record.get("benchmark_summary")),
                median=_fmt(median(speedups) if speedups else None),
                skipped=skipped,
            )
        )

    lines.extend(["", "## Best Speedup Per Task", ""])
    lines.extend(
        [
            "| Task | Overall best | Best run | Candidate path | Speedup vs torch.compile | Reached eager | Recommendation |",
            "| --- | ---: | --- | --- | ---: | --- | --- |",
        ]
    )
    for task_id in tasks:
        best_label = None
        best_record = None
        for label, task_best in best_by_run.items():
            record = task_best.get(task_id)
            if not record:
                continue
            if best_record is None or _metric(record, "speedup_vs_eager") > _metric(best_record, "speedup_vs_eager"):
                best_label = label
                best_record = record
        benchmark = (best_record or {}).get("benchmark_summary") or {}
        speedup = benchmark.get("speedup_vs_eager")
        lines.append(
            "| {task} | {speedup} | {run} | `{path}` | {compile} | {reached} | {recommendation} |".format(
                task=task_id,
                speedup=_fmt(speedup),
                run=best_label or "n/a",
                path=(best_record or {}).get("candidate_path", "n/a"),
                compile=_fmt(benchmark.get("speedup_vs_torch_compile")),
                reached="yes" if speedup is not None and float(speedup) >= 1.0 else "no",
                recommendation=_recommendation(task_id, speedup),
            )
        )

    lines.extend(["", "## Repeatability Summary", ""])
    if repeatability:
        rows = repeatability.get("results") or []
        for row in rows:
            stats = row.get("stats") or {}
            lines.append(
                f"- {row.get('task_id')} `{row.get('candidate_id')}`: median "
                f"{_fmt(stats.get('median'))}x, cv {_fmt(stats.get('coefficient_of_variation'))}, "
                f"stable={'yes' if row.get('stable') else 'no'}"
            )
    else:
        lines.append("- No repeatability_results.json found for the clean focused run.")

    lines.extend(
        [
            "",
            "## Invalid Variant Lesson",
            "",
            "- Non-power-of-two BLOCK_SIZE values are invalid for these templates because they use `tl.arange(0, BLOCK_SIZE)`.",
            "- Invalid template variants should be filtered before verifier/benchmark runs, not counted as model or template compile failures.",
            "",
            "## Conclusion",
            "",
            "- bias_relu is the first real above-eager win in the single-run leaderboard, but the clean-focused repeatability check should be used before treating that as stable.",
            "- vector_add and relu remain poor targets for further standalone optimization because PyTorch eager overhead is already very low on the tested shapes.",
            "- The next useful step is moving to a small fused 8-task set while carrying forward the template validation, repeatability checks, and leaderboard artifacts.",
            "- No SOTA claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def _best_by_task(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        speedup = _metric(record, "speedup_vs_eager")
        if speedup is None:
            continue
        task_id = str(record.get("task_id"))
        if task_id not in best or speedup > _metric(best[task_id], "speedup_vs_eager"):
            best[task_id] = record
    return best


def _speedups(records: list[dict[str, Any]]) -> list[float]:
    return [
        value
        for record in records
        if (value := _metric(record, "speedup_vs_eager")) is not None
    ]


def _metric(record: dict[str, Any], metric: str) -> float | None:
    value = (record.get("benchmark_summary") or {}).get(metric)
    return float(value) if value is not None else None


def _load_repeatability(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "repeatability_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _recommendation(task_id: str, speedup: Any) -> str:
    if speedup is not None and float(speedup) >= 1.0:
        return "useful for dataset; validate with fused-task context"
    if task_id in {"vector_add", "relu"}:
        return "stop standalone optimization; move to fused tasks"
    return "keep as reference, but prioritize fused tasks"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
