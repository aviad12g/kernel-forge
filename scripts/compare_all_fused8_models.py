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

from openkernelforge.reports.fused8 import FUSED8_TASKS
from openkernelforge.reports.run_data import load_run_bundle


DEFAULT_RUNS = {
    "template": "runs/20260519_213349",
    "gemini": "runs/20260519_215314",
    "gemini_guided": "runs/20260519_215439",
    "openai_mini": "runs/20260520_083300",
    "gpt55": "runs/20260520_085334",
}

COMMANDS = {
    "template": "python scripts/run_gpu_baseline_3tasks.py --config configs/template_fused8_gpu_autotune_wide.yaml --out-name template_fused8_gpu_wide",
    "gemini": "python scripts/run_gpu_baseline_3tasks.py --config configs/gemini_fused8_gpu_baseline.yaml --out-name gemini_fused8_gpu_baseline",
    "gemini_guided": "python scripts/run_gpu_baseline_3tasks.py --config configs/gemini_fused8_gpu_template_guided.yaml --out-name gemini_fused8_gpu_template_guided",
    "openai_mini": "python scripts/run_gpu_baseline_3tasks.py --config configs/openai_mini_fused8_gpu_baseline_cheap.yaml --out-name openai_mini_fused8_cheap",
    "gpt55": "python scripts/run_gpu_baseline_3tasks.py --config configs/openai_gpt55_fused8_gpu_baseline_cheap.yaml --out-name openai_gpt55_fused8_cheap",
    "qwen": "python scripts/run_local_model_fused8.py --config configs/qwen_fused8_gpu_baseline_cheap.yaml --out-name qwen_fused8_cheap",
    "deepseek": "python scripts/run_local_model_fused8.py --config configs/deepseek_fused8_gpu_baseline_cheap.yaml --out-name deepseek_fused8_cheap",
    "nemotron": "python scripts/run_local_model_fused8.py --config configs/nemotron_fused8_gpu_baseline_cheap.yaml --out-name nemotron_fused8_cheap",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare all available fused8 model/template runs.")
    parser.add_argument("--template", default=DEFAULT_RUNS["template"])
    parser.add_argument("--gemini", default=DEFAULT_RUNS["gemini"])
    parser.add_argument("--gemini-guided", default=DEFAULT_RUNS["gemini_guided"])
    parser.add_argument("--openai-mini", default=DEFAULT_RUNS["openai_mini"])
    parser.add_argument("--gpt55", default=DEFAULT_RUNS["gpt55"])
    parser.add_argument("--qwen")
    parser.add_argument("--deepseek")
    parser.add_argument("--nemotron")
    parser.add_argument("--out", default="runs/fused8_all_model_comparison.md")
    args = parser.parse_args(argv)

    requested = {
        "template": args.template,
        "gemini": args.gemini,
        "gemini_guided": args.gemini_guided,
        "openai_mini": args.openai_mini,
        "gpt55": args.gpt55,
        "qwen": args.qwen,
        "deepseek": args.deepseek,
        "nemotron": args.nemotron,
    }
    paths = {
        label: Path(path)
        for label, path in requested.items()
        if path and Path(path).exists()
    }
    missing = [label for label, path in requested.items() if path and not Path(path).exists()]
    text = compare_all_models_markdown(paths, missing=missing)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    return 0


def compare_all_models_markdown(paths: dict[str, Path], *, missing: list[str] | None = None) -> str:
    rows = {label: _run_row(path) for label, path in paths.items()}
    lines = [
        "# Fused8 All-Model Comparison",
        "",
        "Internal OpenKernelForge fused8 benchmark only. Not KernelBench and not a SOTA claim.",
        "",
    ]
    if not rows:
        lines.extend(["No requested run directories were found.", "", "## Commands", ""])
        for label, command in COMMANDS.items():
            lines.append(f"- {label}: `{command}`")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| Run | Model | Candidates | Model calls | Cost class | Verify rate | Policy fail rate | Median eager | Median compile | Tasks > eager | Stable tasks > eager | Dataset counts | Worth data gen? |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for label, row in rows.items():
        lines.append(
            "| {label} | `{model}` | {candidates} | {calls} | {cost} | {verify} | {policy} | {median} | {compile} | {tasks} | {stable} | `{dataset}` | {worth} |".format(
                label=label,
                model=row["model"],
                candidates=row["candidates"],
                calls=row["model_calls"],
                cost=row["cost_class"],
                verify=_fmt(row["verification_rate"]),
                policy=_fmt(row["policy_fail_rate"]),
                median=_fmt(row["median_speedup"]),
                compile=_fmt(row["median_compile_speedup"]),
                tasks=row["tasks_gt_eager"],
                stable=row["stable_tasks_gt_eager"],
                dataset=json.dumps(row["dataset_counts"], sort_keys=True),
                worth=_worth_using(row),
            )
        )

    lines.extend(["", "## Best Single-Run Speedup By Task", ""])
    lines.append("| Task | " + " | ".join(rows) + " | Best run |")
    lines.append("| --- | " + " | ".join("---:" for _ in rows) + " | --- |")
    for task in FUSED8_TASKS:
        best_label = None
        best_value = None
        cells = []
        for label, row in rows.items():
            value = row["best_by_task"].get(task)
            cells.append(_fmt(value))
            if value is not None and (best_value is None or value > best_value):
                best_value = value
                best_label = label
        lines.append(f"| {task} | " + " | ".join(cells) + f" | {best_label or 'none'} |")

    lines.extend(["", "## Best Repeat-Stable Speedup By Task", ""])
    lines.append("| Task | " + " | ".join(rows) + " | Stable winner |")
    lines.append("| --- | " + " | ".join("---:" for _ in rows) + " | --- |")
    for task in FUSED8_TASKS:
        best_label = None
        best_value = None
        cells = []
        for label, row in rows.items():
            value = row["repeat_best_by_task"].get(task)
            cells.append(_fmt(value))
            if value is not None and value >= 1.0 and (best_value is None or value > best_value):
                best_value = value
                best_label = label
        lines.append(f"| {task} | " + " | ".join(cells) + f" | {best_label or 'none'} |")

    if missing:
        lines.extend(["", "## Missing Runs", ""])
        for label in missing:
            lines.append(f"- {label}: `{COMMANDS.get(label, 'provide --' + label.replace('_', '-') + ' runs/<run>')}`")
    return "\n".join(lines) + "\n"


def _run_row(path: Path) -> dict[str, Any]:
    bundle = load_run_bundle(path)
    candidates = [record for record in bundle["candidate_records"] if record.get("candidate_path")]
    speedups = _metrics(candidates, "speedup_vs_eager")
    compile_speedups = _metrics(candidates, "speedup_vs_torch_compile")
    best_by_task = _best_by_task(candidates, "speedup_vs_eager")
    repeat_best = _repeat_best(path)
    agent = (bundle.get("config") or {}).get("agent") or {}
    agent_type = agent.get("type")
    return {
        "model": agent.get("model") or agent.get("template_family") or agent_type or "n/a",
        "candidates": len(candidates),
        "model_calls": len(candidates) if agent_type == "llm" else 0,
        "cost_class": _cost_class(agent, path),
        "verification_rate": (
            sum(1 for record in candidates if record.get("verification_passed")) / len(candidates)
            if candidates
            else None
        ),
        "policy_fail_rate": (
            sum(1 for record in candidates if record.get("policy_passed") is False) / len(candidates)
            if candidates
            else None
        ),
        "median_speedup": median(speedups) if speedups else None,
        "median_compile_speedup": median(compile_speedups) if compile_speedups else None,
        "tasks_gt_eager": sum(1 for value in best_by_task.values() if value >= 1.0),
        "stable_tasks_gt_eager": sum(1 for value in repeat_best.values() if value >= 1.0),
        "best_by_task": best_by_task,
        "repeat_best_by_task": repeat_best,
        "dataset_counts": _dataset_counts(path),
    }


def _best_by_task(candidates: list[dict[str, Any]], metric: str) -> dict[str, float]:
    best: dict[str, float] = {}
    for record in candidates:
        value = _metric(record, metric)
        if value is None:
            continue
        task = str(record.get("task_id"))
        if task not in best or value > best[task]:
            best[task] = value
    return best


def _repeat_best(path: Path) -> dict[str, float]:
    repeat_path = path / "repeatability_results.json"
    if not repeat_path.exists():
        return {}
    data = json.loads(repeat_path.read_text(encoding="utf-8"))
    best: dict[str, float] = {}
    for row in data.get("results", []):
        stats = row.get("stats") or {}
        value = stats.get("median")
        if value is None or not row.get("stable"):
            continue
        task = str(row.get("task_id"))
        best[task] = max(best.get(task, 0.0), float(value))
    return best


def _metrics(candidates: list[dict[str, Any]], metric: str) -> list[float]:
    return [value for record in candidates if (value := _metric(record, metric)) is not None]


def _metric(record: dict[str, Any], metric: str) -> float | None:
    value = (record.get("benchmark_summary") or {}).get(metric)
    return float(value) if value is not None else None


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


def _cost_class(agent: dict[str, Any], path: Path) -> str:
    backend = str(agent.get("backend") or "")
    provider = str(agent.get("provider") or "")
    model = str(agent.get("model") or "")
    if agent.get("type") == "template":
        return "none"
    if "local" in provider or "localhost" in str(agent.get("base_url") or ""):
        return "local compute"
    if "gemini" in model:
        return "paid api"
    if "gpt" in model or "openai" in provider or "openai" in backend:
        return "paid api"
    return "unknown"


def _worth_using(row: dict[str, Any]) -> str:
    stable = row["stable_tasks_gt_eager"]
    tasks = row["tasks_gt_eager"]
    verify = row["verification_rate"] or 0.0
    if verify < 0.75:
        return "no: correctness weak"
    if stable >= 2:
        return "yes: repeat-stable wins"
    if tasks >= 3:
        return "maybe: repeatability needed"
    return "limited"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
