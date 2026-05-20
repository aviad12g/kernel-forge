from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openkernelforge.reports.failure_taxonomy import classify_candidate_record
from openkernelforge.reports.run_data import load_run_bundle


COMMANDS = {
    "base_template": (
        "python scripts/run_gpu_baseline_3tasks.py "
        "--config configs/template_3tasks_gpu_autotune.yaml "
        "--out-name template_3tasks_gpu_autotune"
    ),
    "wide_template": (
        "python scripts/run_gpu_baseline_3tasks.py "
        "--config configs/template_3tasks_gpu_autotune_wide.yaml "
        "--out-name template_3tasks_gpu_autotune_wide"
    ),
    "shapeaware_template": (
        "python scripts/run_gpu_baseline_3tasks.py "
        "--config configs/template_3tasks_gpu_autotune_shapeaware.yaml "
        "--out-name template_3tasks_gpu_autotune_shapeaware"
    ),
    "template_copy_wide": (
        "python scripts/run_gpu_baseline_3tasks.py "
        "--config configs/gemini_3_1_flash_lite_3tasks_gpu_template_copy_wide.yaml "
        "--out-name gemini_gpu_3tasks_template_copy_wide"
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare template sweep runs.")
    parser.add_argument("--base-template", help="Original template autotune run")
    parser.add_argument("--wide-template", help="Expanded wide template run")
    parser.add_argument("--shapeaware-template", help="Expanded shape-aware template run")
    parser.add_argument("--template-copy-wide", help="Strict template-copy-wide run")
    parser.add_argument("--out", help="Optional Markdown output path")
    args = parser.parse_args(argv)

    requested = {
        "base_template": args.base_template,
        "wide_template": args.wide_template,
        "shapeaware_template": args.shapeaware_template,
        "template_copy_wide": args.template_copy_wide,
    }
    available = {
        label: Path(path)
        for label, path in requested.items()
        if path and Path(path).exists()
    }
    lines: list[str] = []
    missing = [label for label, path in requested.items() if not path or not Path(path).exists()]
    if missing:
        lines.extend(["## Missing Runs", ""])
        for label in missing:
            lines.append(f"- {label}: run with `{COMMANDS[label]}`")
        lines.append("")
    if available:
        lines.append(compare_template_sweeps_markdown(available))
    else:
        lines.append("No available run directories were provided.")
    text = "\n".join(lines)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    return 0


def compare_template_sweeps_markdown(run_paths: dict[str, Path]) -> str:
    rows = {label: _run_row(path) for label, path in run_paths.items()}
    lines = [
        "# Template Sweep Comparison",
        "",
        "| Run | Candidates | Verification pass | Median speedup | >=1.0x | >=0.8x | Best vector_add | Best relu | Best bias_relu |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in rows.items():
        best = row["best_by_task"]
        lines.append(
            "| {label} | {candidates} | {verification} | {median_speedup} | {fast} | {promising} | {vector} | {relu} | {bias} |".format(
                label=label,
                candidates=row["candidates"],
                verification=row["verification"],
                median_speedup=_fmt(row["median_speedup"]),
                fast=row["fast_count"],
                promising=row["promising_count"],
                vector=_fmt(best.get("vector_add", {}).get("speedup")),
                relu=_fmt(best.get("relu", {}).get("speedup")),
                bias=_fmt(best.get("bias_relu", {}).get("speedup")),
            )
        )

    lines.extend(["", "## Best Metadata Per Task", ""])
    for label, row in rows.items():
        lines.append(f"### {label}")
        for task_id, best in sorted(row["best_by_task"].items()):
            metadata = {
                "block_size": best.get("block_size"),
                "num_warps": best.get("num_warps"),
                "num_stages": best.get("num_stages"),
                "contiguous_policy": best.get("contiguous_policy"),
                "output_allocation_policy": best.get("output_allocation_policy"),
                "shape_specialized": best.get("shape_specialized"),
                "feature_dim_mode": best.get("feature_dim_mode"),
                "n_elements_mode": best.get("n_elements_mode"),
            }
            lines.append(
                f"- {task_id}: {_fmt(best.get('speedup'))} `{best.get('candidate_path')}` "
                f"{json.dumps(metadata, sort_keys=True)}"
            )
        lines.append("")

    if "template_copy_wide" in rows:
        copy = rows["template_copy_wide"]["best_by_task"]
        lines.extend(["## Deterministic Runs Versus Template-Copy-Wide", ""])
        lines.extend(["| Deterministic run | Task | Deterministic best | Copy-wide best | Deterministic beats copy-wide | Gap |", "| --- | --- | ---: | ---: | --- | ---: |"])
        for label, row in rows.items():
            if label == "template_copy_wide":
                continue
            for task_id, best in sorted(row["best_by_task"].items()):
                copy_speed = (copy.get(task_id) or {}).get("speedup")
                gap = best.get("speedup") - copy_speed if copy_speed is not None else None
                lines.append(
                    f"| {label} | {task_id} | {_fmt(best.get('speedup'))} | {_fmt(copy_speed)} | "
                    f"{'yes' if gap is not None and gap > 0 else 'no'} | {_fmt(gap)} |"
                )

    if "shapeaware_template" in rows and "wide_template" in rows:
        lines.extend(["", "## Shape-Aware Delta", ""])
        lines.extend(["| Task | Wide best | Shape-aware best | Shape-aware helps | Delta |", "| --- | ---: | ---: | --- | ---: |"])
        wide = rows["wide_template"]["best_by_task"]
        shape = rows["shapeaware_template"]["best_by_task"]
        for task_id in sorted(set(wide) | set(shape)):
            wide_speed = (wide.get(task_id) or {}).get("speedup")
            shape_speed = (shape.get(task_id) or {}).get("speedup")
            delta = shape_speed - wide_speed if shape_speed is not None and wide_speed is not None else None
            lines.append(
                f"| {task_id} | {_fmt(wide_speed)} | {_fmt(shape_speed)} | "
                f"{'yes' if delta is not None and delta > 0 else 'no'} | {_fmt(delta)} |"
            )
    lines.append("")
    return "\n".join(lines)


def _run_row(path: Path) -> dict[str, Any]:
    bundle = load_run_bundle(path)
    candidates = bundle["candidate_records"]
    speedups = [_speedup(record) for record in candidates if _speedup(record) is not None]
    classifications = Counter(classify_candidate_record(record).failure_type for record in candidates)
    return {
        "path": str(path),
        "candidates": len(candidates),
        "verification": _pct(sum(1 for record in candidates if record.get("verification_passed")), len(candidates)),
        "median_speedup": median(speedups) if speedups else None,
        "fast_count": sum(1 for value in speedups if value >= 1.0),
        "promising_count": sum(1 for value in speedups if value >= 0.8),
        "taxonomy": dict(classifications),
        "best_by_task": _best_by_task(candidates),
    }


def _best_by_task(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in candidates:
        speedup = _speedup(record)
        if speedup is None:
            continue
        task_id = str(record.get("task_id"))
        if task_id not in best or speedup > float(best[task_id]["speedup"]):
            best[task_id] = {
                "speedup": speedup,
                "candidate_path": record.get("candidate_path"),
                "block_size": record.get("block_size"),
                "num_warps": record.get("num_warps"),
                "num_stages": record.get("num_stages"),
                "contiguous_policy": record.get("contiguous_policy"),
                "output_allocation_policy": record.get("output_allocation_policy"),
                "shape_specialized": record.get("shape_specialized"),
                "feature_dim_mode": record.get("feature_dim_mode"),
                "n_elements_mode": record.get("n_elements_mode"),
            }
    return best


def _speedup(record: dict[str, Any]) -> float | None:
    value = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
    return float(value) if value is not None else None


def _pct(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{100 * numerator / denominator:.1f}%"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
