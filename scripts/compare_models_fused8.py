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


COMMANDS = {
    "strong": "python scripts/run_strong_model_fused8.py --out-name strong_fused8",
    "strong_guided": "python scripts/run_strong_model_fused8.py --out-name strong_fused8",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare fused8 model runs with repeatability when available.")
    parser.add_argument("--template", default="runs/20260519_213349")
    parser.add_argument("--gemini", default="runs/20260519_215314")
    parser.add_argument("--gemini-guided", default="runs/20260519_215439")
    parser.add_argument("--strong")
    parser.add_argument("--strong-guided")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    labels = {
        "template": args.template,
        "gemini": args.gemini,
        "gemini_guided": args.gemini_guided,
        "strong": args.strong,
        "strong_guided": args.strong_guided,
    }
    paths = {label: Path(path) for label, path in labels.items() if path and Path(path).exists()}
    missing = [label for label, path in labels.items() if path and not Path(path).exists()]
    if not paths:
        text = _missing_text(["strong", "strong_guided"])
        print(text)
        return 0
    text = compare_models_markdown(paths, missing)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    return 0


def compare_models_markdown(paths: dict[str, Path], missing: list[str] | None = None) -> str:
    rows = {label: _run_row(path) for label, path in paths.items()}
    lines = [
        "# Fused8 Model Comparison",
        "",
        "Internal OpenKernelForge fused8 benchmark only. Not KernelBench and not a SOTA claim.",
        "",
        "| Run | Candidates | Verify rate | Policy fail rate | Median eager | Tasks > eager | Repeat-stable tasks > eager | Dataset counts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, row in rows.items():
        lines.append(
            "| {label} | {candidates} | {verify} | {policy} | {median} | {tasks} | {stable_tasks} | `{dataset}` |".format(
                label=label,
                candidates=row["candidates"],
                verify=_fmt(row["verification_rate"]),
                policy=_fmt(row["policy_fail_rate"]),
                median=_fmt(row["median_speedup"]),
                tasks=row["tasks_ge_1_eager"],
                stable_tasks=row["stable_tasks_ge_1_eager"],
                dataset=json.dumps(row["dataset_counts"], sort_keys=True),
            )
        )

    lines.extend(["", "## Best Repeat Median Per Task", ""])
    tasks = sorted({task for row in rows.values() for task in row["repeat_best"] or row["best"]})
    lines.append("| Task | " + " | ".join(f"{label}" for label in rows) + " | Stable winner |")
    lines.append("| --- | " + " | ".join("---:" for _ in rows) + " | --- |")
    for task in tasks:
        winner_label = None
        winner_value = None
        cells = []
        for label, row in rows.items():
            value = row["repeat_best"].get(task) or row["best"].get(task)
            cells.append(_fmt(value))
            if value is not None and value >= 1.0 and (winner_value is None or value > winner_value):
                winner_value = value
                winner_label = label
        lines.append(f"| {task} | " + " | ".join(cells) + f" | {winner_label or 'none'} |")
    if missing:
        lines.extend(["", "## Missing Runs", ""])
        for label in missing:
            if label in COMMANDS:
                lines.append(f"- {label}: `{COMMANDS[label]}`")
            else:
                lines.append(f"- {label}: path not found")
    return "\n".join(lines) + "\n"


def _run_row(path: Path) -> dict[str, Any]:
    bundle = load_run_bundle(path)
    candidates = [record for record in bundle["candidate_records"] if record.get("candidate_path")]
    speedups = [_metric(record, "speedup_vs_eager") for record in candidates]
    speedups = [value for value in speedups if value is not None]
    best = _best_by_task(candidates)
    repeat_best = _repeat_best(path)
    return {
        "candidates": len(candidates),
        "verification_rate": sum(1 for record in candidates if record.get("verification_passed")) / len(candidates) if candidates else None,
        "policy_fail_rate": sum(1 for record in candidates if record.get("policy_passed") is False) / len(candidates) if candidates else None,
        "median_speedup": median(speedups) if speedups else None,
        "tasks_ge_1_eager": sum(1 for value in best.values() if value >= 1.0),
        "stable_tasks_ge_1_eager": sum(1 for value in repeat_best.values() if value >= 1.0),
        "best": best,
        "repeat_best": repeat_best,
        "dataset_counts": _dataset_counts(path),
    }


def _best_by_task(candidates: list[dict[str, Any]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for record in candidates:
        speedup = _metric(record, "speedup_vs_eager")
        if speedup is None:
            continue
        task = str(record.get("task_id"))
        if task not in best or speedup > best[task]:
            best[task] = speedup
    return best


def _repeat_best(path: Path) -> dict[str, float]:
    repeat_path = path / "repeatability_results.json"
    if not repeat_path.exists():
        return {}
    data = json.loads(repeat_path.read_text(encoding="utf-8"))
    best: dict[str, float] = {}
    for row in data.get("results", []):
        stats = row.get("stats") or {}
        median_value = stats.get("median")
        if median_value is None or not row.get("stable"):
            continue
        task = str(row.get("task_id"))
        value = float(median_value)
        if task not in best or value > best[task]:
            best[task] = value
    return best


def _dataset_counts(run_path: Path) -> dict[str, int]:
    for manifest in sorted(Path("datasets").glob("*/manifest.json")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        source = str(data.get("source_run_dir") or "")
        if Path(source).name == run_path.name:
            return data.get("counts_by_file", {})
    return {}


def _metric(record: dict[str, Any], metric: str) -> float | None:
    value = (record.get("benchmark_summary") or {}).get(metric)
    return float(value) if value is not None else None


def _missing_text(labels: list[str]) -> str:
    lines = ["Missing fused8 model runs.", "", "Generate them with:", ""]
    for label in labels:
        lines.extend(["```bash", COMMANDS.get(label, "python scripts/run_strong_model_fused8.py --out-name strong_fused8"), "```", ""])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
