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
    "template": "python scripts/run_gpu_baseline_3tasks.py --config configs/template_fused8_gpu_autotune_wide.yaml --out-name template_fused8_gpu_wide",
    "llm": "python scripts/run_gpu_baseline_3tasks.py --config configs/gemini_fused8_gpu_baseline.yaml --out-name gemini_fused8_gpu_baseline",
    "template_guided": "python scripts/run_gpu_baseline_3tasks.py --config configs/gemini_fused8_gpu_template_guided.yaml --out-name gemini_fused8_gpu_template_guided",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare fused8 template and model runs.")
    parser.add_argument("--template")
    parser.add_argument("--llm")
    parser.add_argument("--template-guided")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    paths = {
        "template": args.template,
        "llm": args.llm,
        "template_guided": args.template_guided,
    }
    available = {label: Path(path) for label, path in paths.items() if path and Path(path).exists()}
    missing = [label for label, path in paths.items() if not path or not Path(path).exists()]
    if not available:
        text = _missing_text(missing)
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        return 0

    text = compare_fused8_markdown(available, missing)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    return 0


def compare_fused8_markdown(paths: dict[str, Path], missing: list[str] | None = None) -> str:
    rows = {label: _run_row(path) for label, path in paths.items()}
    lines = [
        "# Fused8 Comparison",
        "",
        "This compares internal OpenKernelForge fused8 runs. It is not a KernelBench result.",
        "",
        "| Run | Candidates | Verification rate | Compile error rate | Median speedup | >=1.0x eager | >=1.0x compile | Correct fast | Promising | Dataset counts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, row in rows.items():
        lines.append(
            "| {label} | {candidates} | {verify} | {compile_errors} | {median} | {fast_tasks} | {compile_fast_tasks} | {fast_count} | {promising_count} | `{dataset}` |".format(
                label=label,
                candidates=row["candidates"],
                verify=_fmt(row["verification_rate"]),
                compile_errors=_fmt(row["compile_error_rate"]),
                median=_fmt(row["median_speedup"]),
                fast_tasks=row["tasks_ge_1_eager"],
                compile_fast_tasks=row["tasks_ge_1_compile"],
                fast_count=row["correct_fast_count"],
                promising_count=row["promising_count"],
                dataset=json.dumps(row["dataset_counts"], sort_keys=True),
            )
        )
    lines.extend(["", "## Best Per Task", ""])
    tasks = sorted({task for row in rows.values() for task in row["best"]})
    header = "| Task | " + " | ".join(f"{label} best" for label in rows) + " | Best source |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---:" for _ in rows) + " | --- |")
    for task_id in tasks:
        best_label = None
        best_record = None
        cells = []
        for label, row in rows.items():
            record = row["best"].get(task_id)
            value = (record or {}).get("speedup")
            cells.append(_fmt(value))
            if record and (best_record is None or float(record["speedup"]) > float(best_record["speedup"])):
                best_label = label
                best_record = record
        source = f"{best_label}: {(best_record or {}).get('candidate_path', 'n/a')}"
        lines.append(f"| {task_id} | " + " | ".join(cells) + f" | `{source}` |")
    if missing:
        lines.extend(["", "## Missing Runs", ""])
        for label in missing:
            lines.append(f"- {label}: `{COMMANDS[label]}`")
    return "\n".join(lines) + "\n"


def _missing_text(missing: list[str]) -> str:
    labels = missing or list(COMMANDS)
    lines = ["Missing fused8 runs.", "", "Generate them with:", ""]
    for label in labels:
        lines.extend(["```bash", COMMANDS[label], "```", ""])
    return "\n".join(lines)


def _run_row(path: Path) -> dict[str, Any]:
    bundle = load_run_bundle(path)
    candidates = bundle["candidate_records"]
    speedups = [_metric(record, "speedup_vs_eager") for record in candidates]
    speedups = [value for value in speedups if value is not None]
    best = _best_by_task(candidates)
    compile_errors = sum(
        1
        for record in candidates
        if "compile" in str(record.get("failure_reason") or "").lower()
        or "compile" in str((record.get("verification_summary") or {}).get("first_message") or "").lower()
    )
    return {
        "candidates": len(candidates),
        "verification_rate": (
            sum(1 for record in candidates if record.get("verification_passed")) / len(candidates)
            if candidates
            else None
        ),
        "compile_error_rate": compile_errors / len(candidates) if candidates else None,
        "median_speedup": median(speedups) if speedups else None,
        "tasks_ge_1_eager": sum(1 for record in best.values() if record.get("speedup", 0.0) >= 1.0),
        "tasks_ge_1_compile": sum(1 for record in best.values() if (record.get("compile_speedup") or 0.0) >= 1.0),
        "correct_fast_count": sum(1 for value in speedups if value >= 1.0),
        "promising_count": sum(1 for value in speedups if value >= 0.8),
        "best": best,
        "dataset_counts": _dataset_counts(path),
    }


def _best_by_task(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in candidates:
        speedup = _metric(record, "speedup_vs_eager")
        if speedup is None:
            continue
        task_id = str(record.get("task_id"))
        if task_id not in best or speedup > float(best[task_id]["speedup"]):
            best[task_id] = {
                "speedup": speedup,
                "compile_speedup": _metric(record, "speedup_vs_torch_compile"),
                "candidate_path": record.get("candidate_path"),
            }
    return best


def _dataset_counts(run_path: Path) -> dict[str, int]:
    candidates = [Path("datasets") / run_path.name, Path("datasets") / f"{run_path.name}_export"]
    for dataset_dir in candidates:
        manifest = dataset_dir / "manifest.json"
        if manifest.exists():
            return json.loads(manifest.read_text(encoding="utf-8")).get("counts_by_file", {})
    datasets_root = Path("datasets")
    if datasets_root.exists():
        run_name = run_path.name
        run_strings = {str(run_path), str(run_path.resolve()) if run_path.exists() else str(run_path)}
        for manifest in sorted(datasets_root.glob("*/manifest.json")):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            source = str(data.get("source_run_dir") or "")
            if source in run_strings or Path(source).name == run_name:
                return data.get("counts_by_file", {})
    return {}


def _metric(record: dict[str, Any], metric: str) -> float | None:
    value = (record.get("benchmark_summary") or {}).get(metric)
    return float(value) if value is not None else None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
