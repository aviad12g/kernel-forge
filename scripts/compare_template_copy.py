from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openkernelforge.reports.run_data import load_run_bundle


RUN_COMMAND = (
    "python scripts/run_gpu_baseline_3tasks.py \\\n"
    "  --config configs/gemini_3_1_flash_lite_3tasks_gpu_template_copy.yaml \\\n"
    "  --out-name gemini_gpu_3tasks_template_copy"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare template-copy runs against templates.")
    parser.add_argument("--template", required=True, help="Template autotune run directory")
    parser.add_argument("--template-guided", required=True, help="Template-guided LLM run directory")
    parser.add_argument("--template-copy", help="Template-copy run directory")
    parser.add_argument("--out", help="Optional Markdown output path")
    args = parser.parse_args(argv)

    if not args.template_copy:
        text = "Missing template-copy run.\n\nRun it with:\n\n```bash\n" + RUN_COMMAND + "\n```\n"
        print(text)
        if args.out:
            _write(args.out, text)
        return 0

    missing = [path for path in (args.template, args.template_guided, args.template_copy) if not Path(path).exists()]
    if missing:
        text = "Missing run directory/directories:\n" + "\n".join(f"- {path}" for path in missing) + "\n"
        print(text)
        if args.out:
            _write(args.out, text)
        return 0

    template = _summarize(args.template, "template")
    guided = _summarize(args.template_guided, "template_guided")
    copied = _summarize(args.template_copy, "template_copy")
    text = _format(template, guided, copied)
    print(text)
    if args.out:
        _write(args.out, text)
    return 0


def _summarize(run_dir: str, label: str) -> dict[str, Any]:
    bundle = load_run_bundle(run_dir)
    candidates = bundle["candidate_records"]
    by_task = _best_by_task(candidates)
    scores = [
        float(record.get("preserved_template_structure_score"))
        for record in candidates
        if record.get("preserved_template_structure_score") is not None
    ]
    return {
        "label": label,
        "run_dir": str(run_dir),
        "candidates": len(candidates),
        "verification_pass_rate": _rate(sum(1 for r in candidates if r.get("verification_passed")), len(candidates)),
        "correct_fast": sum(
            1 for r in candidates if ((r.get("benchmark_summary") or {}).get("speedup_vs_eager") or 0) >= 1.0
        ),
        "promising": sum(
            1
            for r in candidates
            if 0.8 <= ((r.get("benchmark_summary") or {}).get("speedup_vs_eager") or 0) < 1.0
        ),
        "slow": sum(
            1 for r in candidates if ((r.get("benchmark_summary") or {}).get("speedup_vs_eager") or 0) < 0.8
        ),
        "best_by_task": by_task,
        "preservation_median": median(scores) if scores else None,
        "forbidden_torch_op_count": sum(1 for r in candidates if r.get("extra_torch_ops_detected")),
        "fallback_count": sum(1 for r in candidates if r.get("fallback_detected")),
    }


def _format(*summaries: dict[str, Any]) -> str:
    tasks = sorted({task for summary in summaries for task in summary["best_by_task"]})
    lines = [
        "# Template Copy Comparison",
        "",
        "| Run | Candidates | Verification pass rate | Preservation median | Forbidden torch ops | Fallbacks | Correct fast | Promising | Slow |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| {label} | {candidates} | {pass_rate} | {score} | {torch_ops} | {fallbacks} | {fast} | {promising} | {slow} |".format(
                label=summary["label"],
                candidates=summary["candidates"],
                pass_rate=_fmt_pct(summary["verification_pass_rate"]),
                score=_fmt(summary["preservation_median"]),
                torch_ops=summary["forbidden_torch_op_count"],
                fallbacks=summary["fallback_count"],
                fast=summary["correct_fast"],
                promising=summary["promising"],
                slow=summary["slow"],
            )
        )
    lines.extend(["", "## Best Speedup Per Task", ""])
    header = "| Task | " + " | ".join(summary["label"] for summary in summaries) + " | Template-copy vs template |"
    sep = "| --- | " + " | ".join("---:" for _ in summaries) + " | ---: |"
    lines.extend([header, sep])
    template_best = summaries[0]["best_by_task"]
    for task in tasks:
        values = [summary["best_by_task"].get(task, {}) for summary in summaries]
        copy_gap = None
        if values[-1].get("speedup") is not None and template_best.get(task, {}).get("speedup") is not None:
            copy_gap = float(values[-1]["speedup"]) - float(template_best[task]["speedup"])
        lines.append(
            "| {task} | {values} | {gap} |".format(
                task=task,
                values=" | ".join(_fmt_speedup(value.get("speedup")) for value in values),
                gap=_fmt_speedup(copy_gap),
            )
        )
    lines.extend(["", "## Best Candidate Source Paths", ""])
    for summary in summaries:
        lines.append(f"### {summary['label']}")
        for task, value in sorted(summary["best_by_task"].items()):
            lines.append(f"- {task}: `{value.get('candidate_path')}`")
        lines.append("")
    return "\n".join(lines)


def _best_by_task(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in candidates:
        speedup = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
        if speedup is None:
            continue
        task = str(record.get("task_id"))
        if task not in best or float(speedup) > float(best[task]["speedup"]):
            best[task] = {
                "speedup": float(speedup),
                "candidate_path": record.get("candidate_path"),
            }
    return best


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_speedup(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}x"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _write(path_value: str, text: str) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
