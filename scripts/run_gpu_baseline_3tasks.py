from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openkernelforge.utils.env_probe import (
    TRITON_EXECUTION_OK,
    format_environment_summary,
    probe_environment,
)
from openkernelforge.config import load_config


DEFAULT_CONFIG = Path("configs/gemini_3_1_flash_lite_baseline_3tasks_gpu.yaml")


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a GPU-verified 3-task baseline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="GPU baseline config path")
    parser.add_argument("--out-name", help="Dataset directory name under datasets/")
    args = parser.parse_args(argv)

    config = Path(args.config)
    environment = probe_environment()
    print(format_environment_summary(environment))
    if environment.viability != TRITON_EXECUTION_OK:
        print("")
        print("GPU baseline refused: CUDA + Triton execution is not viable on this machine.")
        print("This avoids treating local environment failures as model failures.")
        return 1

    run_config = load_config(config)
    if run_config.agent.type == "llm":
        check = run_command(
            [sys.executable, "-m", "openkernelforge.cli", "check-backend", "--config", str(config)]
        )
        _print_output(check)
        if check.returncode != 0:
            print("Backend unavailable; GPU baseline was not run.")
            return check.returncode or 1
    else:
        print(f"Backend check skipped for agent type: {run_config.agent.type}")

    run = run_command([sys.executable, "-m", "openkernelforge.cli", "run", "--config", str(config)])
    _print_output(run)
    if run.returncode != 0:
        print("GPU baseline run failed.")
        return run.returncode

    run_dir = _parse_run_dir(run.stdout)
    if run_dir is None:
        print("Could not determine run directory from CLI output.")
        return 1

    analyze = run_command(
        [sys.executable, "-m", "openkernelforge.cli", "analyze-run", "--run-dir", str(run_dir)]
    )
    _print_output(analyze)
    if analyze.returncode != 0:
        return analyze.returncode

    review = run_command(
        [sys.executable, "-m", "openkernelforge.cli", "review-real-run", "--run-dir", str(run_dir)]
    )
    _print_output(review)
    if review.returncode != 0:
        return review.returncode

    dataset_name = args.out_name or f"{config.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_dir = Path("datasets") / dataset_name
    export = run_command(
        [
            sys.executable,
            "-m",
            "openkernelforge.cli",
            "export-dataset",
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(dataset_dir),
        ]
    )
    _print_output(export)
    if export.returncode != 0:
        return export.returncode

    validate = run_command(
        [sys.executable, "-m", "openkernelforge.cli", "validate-dataset", "--dataset-dir", str(dataset_dir)]
    )
    _print_output(validate)
    if validate.returncode != 0:
        return validate.returncode

    manifest = _load_manifest(dataset_dir)
    metadata = _load_json(run_dir / "run_metadata.json")
    print("")
    print("GPU baseline complete.")
    print(f"Environment viability: {environment.viability}")
    print(f"Run directory: {run_dir}")
    print(f"Analysis path: {run_dir / 'analysis.md'}")
    print(f"Review path: {run_dir / 'real_run_review.md'}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Failure taxonomy counts: {manifest.get('counts_by_failure_type', {})}")
    print(f"Dataset counts: {manifest.get('counts_by_file', {})}")
    print(f"Selected correct tasks: {metadata.get('selected_correct_tasks', 'n/a')}")
    print(f"Benchmarked candidates: {metadata.get('benchmarked_candidate_count', 'n/a')}")
    return 0


def _parse_run_dir(output: str) -> Path | None:
    match = re.search(r"Run complete:\s*(\S+)", output)
    return Path(match.group(1)) if match else None


def _load_manifest(dataset_dir: Path) -> dict:
    return _load_json(dataset_dir / "manifest.json")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _print_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")


if __name__ == "__main__":
    raise SystemExit(main())
