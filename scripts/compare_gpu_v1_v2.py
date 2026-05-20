from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from openkernelforge.reports.failure_taxonomy import classify_candidate_record
from openkernelforge.reports.run_data import load_run_bundle


V2_COMMAND = (
    "python scripts/run_gpu_baseline_3tasks.py "
    "--config configs/gemini_3_1_flash_lite_baseline_3tasks_gpu_v2.yaml "
    "--out-name gemini_gpu_3tasks_v2"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare GPU v1 and v2 OpenKernelForge runs.")
    parser.add_argument("--v1", required=True, help="GPU v1 run directory")
    parser.add_argument("--v2", help="GPU v2 run directory")
    parser.add_argument("--out", help="Optional Markdown report path")
    args = parser.parse_args(argv)

    if not args.v2 or not Path(args.v2).exists():
        text = (
            "Missing v2 run.\n\nRun it with:\n\n"
            "```bash\n"
            f"{V2_COMMAND}\n"
            "```\n"
        )
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        return 0

    table = compare_gpu_runs_markdown(args.v1, args.v2)
    print(table, end="")
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
    return 0


def compare_gpu_runs_markdown(v1: str | Path, v2: str | Path) -> str:
    rows = [_run_row(Path(v1), "v1"), _run_row(Path(v2), "v2")]
    lines = [
        "| Run | Dir | Tasks | Candidates | Policy pass | Verification pass | Benchmarked | Selected correct | Median speedup eager | Correct-but-slow | Correct-fast | Compile errors | Repair pairs | Optimization pairs | Rejected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {path} | {tasks} | {candidates} | {policy} | {verification} | {benchmarked} | {selected} | {median_speedup} | {slow} | {fast} | {compile_errors} | {repair_pairs} | {optimization_pairs} | {rejected} |".format(
                **row
            )
        )
    lines.append("")
    lines.append("## Best Speedup Vs Eager Per Task")
    lines.append("")
    lines.append("| Task | v1 | v2 |")
    lines.append("| --- | ---: | ---: |")
    task_ids = sorted(set(rows[0]["best_by_task"]) | set(rows[1]["best_by_task"]))
    for task_id in task_ids:
        lines.append(
            f"| {task_id} | {_fmt(rows[0]['best_by_task'].get(task_id))} | {_fmt(rows[1]['best_by_task'].get(task_id))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _run_row(path: Path, label: str) -> dict[str, Any]:
    bundle = load_run_bundle(path)
    candidates = bundle["candidate_records"]
    task_records = bundle["task_records"]
    classifications = [classify_candidate_record(record) for record in candidates]
    counts = Counter(item.failure_type for item in classifications)
    speedups = [
        float((record.get("benchmark_summary") or {}).get("speedup_vs_eager"))
        for record in candidates
        if (record.get("benchmark_summary") or {}).get("speedup_vs_eager") is not None
    ]
    return {
        "label": label,
        "path": str(path),
        "tasks": len(task_records),
        "candidates": len(candidates),
        "policy": _pct(sum(1 for record in candidates if record.get("policy_passed")), len(candidates)),
        "verification": _pct(sum(1 for record in candidates if record.get("verification_passed")), len(candidates)),
        "benchmarked": sum(1 for record in candidates if record.get("benchmark_summary")),
        "selected": sum(1 for record in candidates if record.get("selected_best") and record.get("verification_passed")),
        "median_speedup": _fmt(median(speedups) if speedups else None),
        "slow": counts.get("CORRECT_BUT_SLOW", 0) + counts.get("CORRECT_PROMISING_BUT_SLOW", 0),
        "fast": counts.get("CORRECT_AND_FAST", 0),
        "compile_errors": counts.get("TRITON_COMPILE_ERROR", 0),
        "repair_pairs": _repair_pair_count(candidates),
        "optimization_pairs": _optimization_pair_count(candidates),
        "rejected": sum(1 for record in candidates if not record.get("verification_passed")),
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


def _repair_pair_count(candidates: list[dict[str, Any]]) -> int:
    count = 0
    for records in _by_task(candidates).values():
        seen_failed = 0
        for record in records:
            if record.get("verification_passed"):
                count += seen_failed
                break
            seen_failed += 1
    return count


def _optimization_pair_count(candidates: list[dict[str, Any]]) -> int:
    count = 0
    for records in _by_task(candidates).values():
        speeds = sorted(
            float((record.get("benchmark_summary") or {}).get("candidate_median_ms"))
            for record in records
            if record.get("verification_passed")
            and (record.get("benchmark_summary") or {}).get("candidate_median_ms") is not None
        )
        if len(speeds) < 2:
            continue
        fastest = speeds[0]
        count += sum(1 for value in speeds[1:] if value > fastest * 1.05)
    return count


def _by_task(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        grouped[str(record.get("task_id"))].append(record)
    for records in grouped.values():
        records.sort(key=lambda r: (r.get("attempt_index") or 0, r.get("candidate_index") or 0))
    return grouped


def _pct(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{100 * numerator / denominator:.1f}%"


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
