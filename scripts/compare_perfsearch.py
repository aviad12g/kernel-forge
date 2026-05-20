from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from openkernelforge.reports.failure_taxonomy import classify_candidate_record
from openkernelforge.reports.run_data import load_run_bundle


PERFSEARCH_COMMAND = (
    "python scripts/run_gpu_baseline_3tasks.py "
    "--config configs/gemini_3_1_flash_lite_3tasks_gpu_perfsearch.yaml "
    "--out-name gemini_gpu_3tasks_perfsearch"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a baseline run to a performance-search run.")
    parser.add_argument("--baseline", required=True, help="Baseline run directory")
    parser.add_argument("--perfsearch", help="Performance-search run directory")
    parser.add_argument("--out", help="Optional Markdown output path")
    args = parser.parse_args(argv)

    if not Path(args.baseline).exists():
        text = f"Missing baseline run: {args.baseline}\n"
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        return 0

    if not args.perfsearch or not Path(args.perfsearch).exists():
        text = (
            "Missing performance-search run.\n\nRun it with:\n\n"
            "```bash\n"
            f"{PERFSEARCH_COMMAND}\n"
            "```\n"
        )
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        return 0

    report = compare_perfsearch_markdown(args.baseline, args.perfsearch)
    print(report, end="")
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    return 0


def compare_perfsearch_markdown(baseline: str | Path, perfsearch: str | Path) -> str:
    rows = [_run_row(Path(baseline), "baseline"), _run_row(Path(perfsearch), "perfsearch")]
    lines = [
        "| Run | Dir | Candidates | Verified | Benchmarked | Median speedup | Correct fast | Promising | Slow | Target reached | Optimization pairs | Generated perf candidates | Candidates per target |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {path} | {candidates} | {verified} | {benchmarked} | {median_speedup} | {fast} | {promising} | {slow} | {target_reached} | {optimization_pairs} | {perf_candidates} | {cost_proxy} |".format(
                **row
            )
        )
    lines.extend(["", "## Best Speedup Vs Eager Per Task", "", "| Task | baseline | perfsearch | Delta |", "| --- | ---: | ---: | ---: |"])
    task_ids = sorted(set(rows[0]["best_by_task"]) | set(rows[1]["best_by_task"]))
    for task_id in task_ids:
        base = rows[0]["best_by_task"].get(task_id)
        perf = rows[1]["best_by_task"].get(task_id)
        delta = perf - base if base is not None and perf is not None else None
        lines.append(f"| {task_id} | {_fmt(base)} | {_fmt(perf)} | {_fmt(delta)} |")
    lines.append("")
    return "\n".join(lines)


def _run_row(path: Path, label: str) -> dict[str, Any]:
    bundle = load_run_bundle(path)
    candidates = bundle["candidate_records"]
    classifications = [classify_candidate_record(record) for record in candidates]
    counts = Counter(item.failure_type for item in classifications)
    speedups = [
        float((record.get("benchmark_summary") or {}).get("speedup_vs_eager"))
        for record in candidates
        if (record.get("benchmark_summary") or {}).get("speedup_vs_eager") is not None
    ]
    target_reached_tasks = {
        record.get("task_id") for record in candidates if record.get("target_reached")
    }
    perf_candidates = [record for record in candidates if record.get("generation_stage") == "performance_search"]
    target_count = len(target_reached_tasks)
    return {
        "label": label,
        "path": str(path),
        "candidates": len(candidates),
        "verified": sum(1 for record in candidates if record.get("verification_passed")),
        "benchmarked": sum(1 for record in candidates if record.get("benchmark_summary")),
        "median_speedup": _fmt(median(speedups) if speedups else None),
        "fast": counts.get("CORRECT_AND_FAST", 0),
        "promising": counts.get("CORRECT_PROMISING_BUT_SLOW", 0),
        "slow": counts.get("CORRECT_BUT_SLOW", 0),
        "target_reached": target_count,
        "optimization_pairs": _optimization_pair_count(candidates),
        "perf_candidates": len(perf_candidates),
        "cost_proxy": _fmt(len(perf_candidates) / target_count if target_count else None),
        "best_by_task": _best_by_task(candidates),
    }


def _best_by_task(candidates: list[dict[str, Any]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for record in candidates:
        speedup = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
        if speedup is None:
            continue
        task_id = str(record.get("task_id"))
        best[task_id] = max(best.get(task_id, float("-inf")), float(speedup))
    return best


def _optimization_pair_count(candidates: list[dict[str, Any]]) -> int:
    return sum(
        1
        for record in candidates
        if record.get("generation_stage") == "performance_search"
        and record.get("verification_passed")
        and record.get("improved_over_parent")
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
