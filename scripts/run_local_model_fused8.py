from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openkernelforge.config import load_config
from scripts.check_local_model_server import check_local_model_server


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one cheap fused8 baseline against a local model server.")
    parser.add_argument("--config", default="configs/qwen_fused8_gpu_baseline_cheap.yaml")
    parser.add_argument("--out-name", default="qwen_fused8_cheap")
    parser.add_argument("--template-run", default="runs/20260519_213349")
    parser.add_argument("--gemini-run", default="runs/20260519_215314")
    parser.add_argument("--gemini-guided-run", default="runs/20260519_215439")
    parser.add_argument("--openai-mini-run", default="runs/20260520_083300")
    parser.add_argument("--gpt55-run", default="runs/20260520_085334")
    args = parser.parse_args(argv)

    env = run_command([sys.executable, "-m", "openkernelforge.cli", "env-check"])
    _print_output(env)
    if env.returncode != 0 or "TRITON_EXECUTION_OK" not in env.stdout:
        print("GPU environment is not viable; refusing local fused8 model run.")
        return env.returncode or 1

    config = load_config(args.config)
    server = check_local_model_server(
        base_url=config.agent.base_url or "http://localhost:8000/v1",
        model=config.agent.model,
        api_key_env=config.agent.api_key_env,
        timeout=min(float(config.agent.timeout_seconds), 30.0),
    )
    if not server.available:
        print("Local model server unavailable; fused8 run was not started.")
        print(server.message)
        print("")
        print("Start a local OpenAI-compatible server, for example:")
        print("python -m vllm.entrypoints.openai.api_server --model <model_name_or_path> --host 0.0.0.0 --port 8000")
        return 1
    print(server.message)
    print(f"Using local model: {server.model}")

    run = run_command(
        [
            sys.executable,
            "scripts/run_gpu_baseline_3tasks.py",
            "--config",
            args.config,
            "--out-name",
            args.out_name,
        ]
    )
    _print_output(run)
    if run.returncode != 0:
        return run.returncode
    run_dir = _parse_run_dir(run.stdout)
    if run_dir is None:
        print("Could not determine run directory from output.")
        return 1

    fused8 = run_command([sys.executable, "-m", "openkernelforge.cli", "fused8-report", "--run-dir", str(run_dir)])
    _print_output(fused8)
    if fused8.returncode != 0:
        return fused8.returncode

    metrics = _run_metrics(run_dir)
    if metrics["tasks_gt_eager"] > 0:
        repeat = run_command(
            [
                sys.executable,
                "-m",
                "openkernelforge.cli",
                "repeatability-report",
                "--run-dir",
                str(run_dir),
                "--top-k",
                "1",
                "--repeats",
                "3",
            ]
        )
        _print_output(repeat)
        if repeat.returncode != 0:
            return repeat.returncode

    compare = run_command(
        [
            sys.executable,
            "scripts/compare_all_fused8_models.py",
            "--template",
            args.template_run,
            "--gemini",
            args.gemini_run,
            "--gemini-guided",
            args.gemini_guided_run,
            "--openai-mini",
            args.openai_mini_run,
            "--gpt55",
            args.gpt55_run,
            "--qwen",
            str(run_dir),
            "--out",
            "runs/fused8_all_model_comparison.md",
        ]
    )
    _print_output(compare)
    if compare.returncode != 0:
        return compare.returncode

    metrics = _run_metrics(run_dir)
    print("")
    print("Local fused8 model run complete.")
    print(f"Run directory: {run_dir}")
    print(f"Candidates: {metrics['candidates']}")
    print(f"Verification rate: {_fmt(metrics['verification_rate'])}")
    print(f"Median speedup vs eager: {_fmt(metrics['median_speedup'])}")
    print(f"Tasks beating eager: {metrics['tasks_gt_eager']}")
    print(f"Repeat-stable wins: {metrics['stable_tasks_gt_eager']}")
    print(f"Dataset counts: {metrics['dataset_counts']}")
    return 0


def _run_metrics(run_dir: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = [row for row in rows if row.get("record_type") == "candidate"]
    speedups = [
        float(value)
        for row in candidates
        if (value := (row.get("benchmark_summary") or {}).get("speedup_vs_eager")) is not None
    ]
    best_by_task: dict[str, float] = {}
    for row in candidates:
        value = (row.get("benchmark_summary") or {}).get("speedup_vs_eager")
        if value is None:
            continue
        task = str(row.get("task_id"))
        best_by_task[task] = max(best_by_task.get(task, 0.0), float(value))
    return {
        "candidates": len(candidates),
        "verification_rate": (
            sum(1 for row in candidates if row.get("verification_passed")) / len(candidates)
            if candidates
            else None
        ),
        "median_speedup": median(speedups) if speedups else None,
        "tasks_gt_eager": sum(1 for value in best_by_task.values() if value >= 1.0),
        "stable_tasks_gt_eager": len(_repeat_stable_wins(run_dir)),
        "dataset_counts": _dataset_counts(run_dir),
    }


def _repeat_stable_wins(run_dir: Path) -> dict[str, float]:
    path = run_dir / "repeatability_results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    wins: dict[str, float] = {}
    for row in data.get("results", []):
        value = ((row.get("stats") or {}).get("median"))
        if value is not None and row.get("stable") and float(value) >= 1.0:
            wins[str(row.get("task_id"))] = float(value)
    return wins


def _dataset_counts(run_dir: Path) -> dict[str, int]:
    for manifest in sorted(Path("datasets").glob("*/manifest.json")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if Path(str(data.get("source_run_dir") or "")).name == run_dir.name:
            return data.get("counts_by_file", {})
    return {}


def _parse_run_dir(output: str) -> Path | None:
    match = re.search(r"Run directory:\s*(\S+)", output) or re.search(r"Run complete:\s*(\S+)", output)
    return Path(match.group(1)) if match else None


def _print_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
