from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from openkernelforge.reports.failure_taxonomy import classify_candidate_record
from openkernelforge.reports.run_data import load_run_bundle


TEMPLATE_COMMAND = (
    "python scripts/run_gpu_baseline_3tasks.py "
    "--config configs/template_3tasks_gpu_autotune.yaml "
    "--out-name template_3tasks_gpu_autotune"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare an LLM run against deterministic templates.")
    parser.add_argument("--llm", required=True, help="LLM run directory")
    parser.add_argument("--template", help="Template autotune run directory")
    parser.add_argument("--out", help="Optional Markdown output path")
    args = parser.parse_args(argv)

    if not args.template or not Path(args.template).exists():
        text = (
            "Missing template run.\n\nRun it with:\n\n"
            "```bash\n"
            f"{TEMPLATE_COMMAND}\n"
            "```\n"
        )
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        return 0

    report = compare_llm_vs_templates_markdown(args.llm, args.template)
    print(report, end="")
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    return 0


def compare_llm_vs_templates_markdown(llm_run: str | Path, template_run: str | Path) -> str:
    rows = [_run_row(Path(llm_run), "llm"), _run_row(Path(template_run), "template")]
    lines = [
        "| Run | Dir | Candidates | Verification pass | Median speedup | Correct fast | Promising | Slow |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {path} | {candidates} | {verification} | {median_speedup} | {fast} | {promising} | {slow} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Best Speedup Vs Eager Per Task",
            "",
            "| Task | LLM best | Template best | Template beats LLM | Gap | Template source |",
            "| --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    task_ids = sorted(set(rows[0]["best_by_task"]) | set(rows[1]["best_by_task"]))
    for task_id in task_ids:
        llm = rows[0]["best_by_task"].get(task_id)
        template = rows[1]["best_by_task"].get(task_id)
        gap = template - llm if template is not None and llm is not None else None
        template_record = rows[1]["best_record_by_task"].get(task_id, {})
        lines.append(
            f"| {task_id} | {_fmt(llm)} | {_fmt(template)} | "
            f"{'yes' if gap is not None and gap > 0 else 'no'} | {_fmt(gap)} | "
            f"`{template_record.get('candidate_path', 'n/a')}` |"
        )
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
    return {
        "label": label,
        "path": str(path),
        "candidates": len(candidates),
        "verification": _pct(sum(1 for record in candidates if record.get("verification_passed")), len(candidates)),
        "median_speedup": _fmt(median(speedups) if speedups else None),
        "fast": counts.get("CORRECT_AND_FAST", 0),
        "promising": counts.get("CORRECT_PROMISING_BUT_SLOW", 0),
        "slow": counts.get("CORRECT_BUT_SLOW", 0),
        "best_by_task": _best_by_task(candidates),
        "best_record_by_task": _best_record_by_task(candidates),
    }


def _best_by_task(candidates: list[dict[str, Any]]) -> dict[str, float]:
    return {
        task_id: float((record.get("benchmark_summary") or {}).get("speedup_vs_eager"))
        for task_id, record in _best_record_by_task(candidates).items()
    }


def _best_record_by_task(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in candidates:
        speedup = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
        if speedup is None:
            continue
        task_id = str(record.get("task_id"))
        current = best.get(task_id)
        current_speedup = (
            (current.get("benchmark_summary") or {}).get("speedup_vs_eager")
            if current
            else None
        )
        if current is None or float(speedup) > float(current_speedup):
            best[task_id] = record
    return best


def _pct(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{100 * numerator / denominator:.1f}%"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
