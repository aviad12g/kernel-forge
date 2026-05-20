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


FOCUSED_COMMAND = (
    "python scripts/run_gpu_baseline_3tasks.py "
    "--config configs/template_3tasks_gpu_autotune_focused.yaml "
    "--out-name template_3tasks_gpu_autotune_focused"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare focused template sweep results.")
    parser.add_argument("--shapeaware", required=True, help="Shape-aware template run")
    parser.add_argument("--template-copy-wide", required=True, help="Template-copy-wide run")
    parser.add_argument("--focused", help="Focused sweep run")
    parser.add_argument("--out", help="Optional Markdown output path")
    args = parser.parse_args(argv)

    if not args.focused or not Path(args.focused).exists():
        text = (
            "Missing focused sweep run.\n\nRun it with:\n\n"
            "```bash\n"
            f"{FOCUSED_COMMAND}\n"
            "```\n"
        )
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        return 0

    text = compare_focused_sweep_markdown(args.shapeaware, args.template_copy_wide, args.focused)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    return 0


def compare_focused_sweep_markdown(
    shapeaware: str | Path,
    template_copy_wide: str | Path,
    focused: str | Path,
) -> str:
    rows = {
        "shapeaware": _run_row(Path(shapeaware)),
        "template_copy_wide": _run_row(Path(template_copy_wide)),
        "focused": _run_row(Path(focused)),
    }
    lines = [
        "# Focused Sweep Comparison",
        "",
        "| Run | Candidates | Median eager speedup | >=1.0x eager | >=0.95x eager | >=0.8x eager | >=1.0x compile |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in rows.items():
        lines.append(
            "| {label} | {candidates} | {median} | {fast} | {near} | {promising} | {compile_fast} |".format(
                label=label,
                candidates=row["candidates"],
                median=_fmt(row["median_speedup"]),
                fast=row["fast_count"],
                near=row["near_eager_count"],
                promising=row["promising_count"],
                compile_fast=row["compile_fast_count"],
            )
        )
    lines.extend(["", "## Best Per Task", ""])
    lines.extend(
        [
            "| Task | Shape-aware eager | Copy-wide eager | Focused eager | Focused compile | Improved previous best | Recommendation |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    tasks = sorted(set(rows["shapeaware"]["best"]) | set(rows["template_copy_wide"]["best"]) | set(rows["focused"]["best"]))
    for task_id in tasks:
        shape = rows["shapeaware"]["best"].get(task_id, {})
        copy = rows["template_copy_wide"]["best"].get(task_id, {})
        focus = rows["focused"]["best"].get(task_id, {})
        previous = max(
            value
            for value in (shape.get("eager"), copy.get("eager"))
            if value is not None
        )
        improved = focus.get("eager") is not None and focus["eager"] > previous
        lines.append(
            "| {task} | {shape} | {copy} | {focused} | {compile} | {improved} | {recommendation} |".format(
                task=task_id,
                shape=_fmt(shape.get("eager")),
                copy=_fmt(copy.get("eager")),
                focused=_fmt(focus.get("eager")),
                compile=_fmt(focus.get("compile")),
                improved="yes" if improved else "no",
                recommendation=_recommendation(focus.get("eager"), focus.get("compile"), previous),
            )
        )
    lines.extend(["", "## Focused Best Metadata", ""])
    for task_id, best in sorted(rows["focused"]["best"].items()):
        metadata = {key: best.get(key) for key in (
            "candidate_path",
            "block_size",
            "num_warps",
            "num_stages",
            "contiguous_policy",
            "output_allocation_policy",
            "n_elements_mode",
            "feature_dim_mode",
            "shape_specialized",
        )}
        lines.append(f"- {task_id}: {json.dumps(metadata, sort_keys=True)}")
    return "\n".join(lines) + "\n"


def _run_row(path: Path) -> dict[str, Any]:
    bundle = load_run_bundle(path)
    candidates = bundle["candidate_records"]
    eager_values = [_metric(record, "speedup_vs_eager") for record in candidates]
    eager_values = [value for value in eager_values if value is not None]
    compile_values = [_metric(record, "speedup_vs_torch_compile") for record in candidates]
    compile_values = [value for value in compile_values if value is not None]
    return {
        "candidates": len(candidates),
        "median_speedup": median(eager_values) if eager_values else None,
        "fast_count": sum(1 for value in eager_values if value >= 1.0),
        "near_eager_count": sum(1 for value in eager_values if value >= 0.95),
        "promising_count": sum(1 for value in eager_values if value >= 0.8),
        "compile_fast_count": sum(1 for value in compile_values if value >= 1.0),
        "best": _best_by_task(candidates),
    }


def _best_by_task(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in candidates:
        eager = _metric(record, "speedup_vs_eager")
        if eager is None:
            continue
        task_id = str(record.get("task_id"))
        if task_id not in best or eager > best[task_id]["eager"]:
            best[task_id] = {
                "eager": eager,
                "compile": _metric(record, "speedup_vs_torch_compile"),
                "candidate_path": record.get("candidate_path"),
                "block_size": record.get("block_size"),
                "num_warps": record.get("num_warps"),
                "num_stages": record.get("num_stages"),
                "contiguous_policy": record.get("contiguous_policy"),
                "output_allocation_policy": record.get("output_allocation_policy"),
                "n_elements_mode": record.get("n_elements_mode"),
                "feature_dim_mode": record.get("feature_dim_mode"),
                "shape_specialized": record.get("shape_specialized"),
            }
    return best


def _metric(record: dict[str, Any], metric: str) -> float | None:
    value = (record.get("benchmark_summary") or {}).get(metric)
    return float(value) if value is not None else None


def _recommendation(focused: float | None, compile_speed: float | None, previous: float | None) -> str:
    if focused is not None and focused >= 1.0:
        return "include in dataset and validate with profiler"
    if compile_speed is not None and compile_speed >= 1.0:
        return "beats torch.compile baseline; inspect source"
    if focused is not None and previous is not None and focused > previous:
        return "continue optimizing this task"
    return "add profiler or move to fused tasks"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
