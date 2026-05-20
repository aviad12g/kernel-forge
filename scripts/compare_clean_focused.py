from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openkernelforge.reports.run_data import load_run_bundle
from openkernelforge.reports.skipped_variants import load_skipped_variants


CLEAN_COMMAND = (
    "python scripts/run_gpu_baseline_3tasks.py "
    "--config configs/template_3tasks_gpu_autotune_focused_clean.yaml "
    "--out-name template_3tasks_gpu_autotune_focused_clean"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare old and clean focused template sweeps.")
    parser.add_argument("--shapeaware", required=True)
    parser.add_argument("--template-copy-wide", required=True)
    parser.add_argument("--focused", required=True)
    parser.add_argument("--clean-focused")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    if not args.clean_focused or not Path(args.clean_focused).exists():
        text = (
            "Missing clean focused sweep run.\n\nRun it with:\n\n"
            "```bash\n"
            f"{CLEAN_COMMAND}\n"
            "```\n"
        )
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        return 0

    text = compare_markdown(
        {
            "shapeaware": args.shapeaware,
            "template_copy_wide": args.template_copy_wide,
            "focused": args.focused,
            "clean_focused": args.clean_focused,
        }
    )
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    return 0


def compare_markdown(paths: dict[str, str | Path]) -> str:
    rows = {label: _run_row(Path(path)) for label, path in paths.items()}
    lines = [
        "# Clean Focused Sweep Comparison",
        "",
        "| Run | Candidates | Skipped | Compile errors | Verification pass rate | Median speedup | >=1.0x | >=0.95x | >=0.8x |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in rows.items():
        lines.append(
            "| {label} | {candidates} | {skipped} | {compile_errors} | {verify} | {median} | {fast} | {near} | {promising} |".format(
                label=label,
                candidates=row["candidates"],
                skipped=row["skipped"],
                compile_errors=row["compile_errors"],
                verify=_fmt(row["verification_pass_rate"]),
                median=_fmt(row["median_speedup"]),
                fast=row["fast_count"],
                near=row["near_eager_count"],
                promising=row["promising_count"],
            )
        )
    lines.extend(["", "## Best Per Task", ""])
    lines.extend(["| Task | Shape-aware | Copy-wide | Focused | Clean focused | Best metadata |", "| --- | ---: | ---: | ---: | ---: | --- |"])
    tasks = sorted({task for row in rows.values() for task in row["best"]})
    for task_id in tasks:
        clean_best = rows["clean_focused"]["best"].get(task_id, {})
        metadata = {
            key: clean_best.get(key)
            for key in (
                "candidate_path",
                "block_size",
                "num_warps",
                "num_stages",
                "contiguous_policy",
                "output_allocation_policy",
                "n_elements_mode",
                "feature_dim_mode",
            )
        }
        lines.append(
            "| {task} | {shape} | {copy} | {focused} | {clean} | `{metadata}` |".format(
                task=task_id,
                shape=_fmt((rows["shapeaware"]["best"].get(task_id) or {}).get("speedup")),
                copy=_fmt((rows["template_copy_wide"]["best"].get(task_id) or {}).get("speedup")),
                focused=_fmt((rows["focused"]["best"].get(task_id) or {}).get("speedup")),
                clean=_fmt(clean_best.get("speedup")),
                metadata=json.dumps(metadata, sort_keys=True),
            )
        )
    return "\n".join(lines) + "\n"


def _run_row(path: Path) -> dict[str, Any]:
    bundle = load_run_bundle(path)
    candidates = bundle["candidate_records"]
    speedups = [
        float(value)
        for record in candidates
        if (value := (record.get("benchmark_summary") or {}).get("speedup_vs_eager")) is not None
    ]
    compile_errors = sum(
        1
        for record in candidates
        if "compile" in str(record.get("failure_reason") or "").lower()
        or "compile" in str((record.get("verification_summary") or {}).get("first_message") or "").lower()
    )
    return {
        "candidates": len(candidates),
        "skipped": len(load_skipped_variants(path)),
        "compile_errors": compile_errors,
        "verification_pass_rate": (
            sum(1 for record in candidates if record.get("verification_passed")) / len(candidates)
            if candidates
            else None
        ),
        "median_speedup": median(speedups) if speedups else None,
        "fast_count": sum(1 for value in speedups if value >= 1.0),
        "near_eager_count": sum(1 for value in speedups if value >= 0.95),
        "promising_count": sum(1 for value in speedups if value >= 0.8),
        "best": _best_by_task(candidates),
    }


def _best_by_task(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in candidates:
        value = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
        if value is None:
            continue
        task_id = str(record.get("task_id"))
        speedup = float(value)
        if task_id not in best or speedup > best[task_id]["speedup"]:
            best[task_id] = {**record, "speedup": speedup}
    return best


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
