from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a stronger model fused8 baseline and template-guided pass.")
    parser.add_argument("--baseline-config", default="configs/strong_model_fused8_gpu_baseline.yaml")
    parser.add_argument("--guided-config", default="configs/strong_model_fused8_gpu_template_guided.yaml")
    parser.add_argument("--template-run", default="runs/20260519_213349")
    parser.add_argument("--gemini-run", default="runs/20260519_215314")
    parser.add_argument("--gemini-guided-run", default="runs/20260519_215439")
    parser.add_argument("--out-name", default="strong_fused8")
    args = parser.parse_args(argv)

    env = run_command([sys.executable, "-m", "openkernelforge.cli", "env-check"])
    _print_output(env)
    if env.returncode != 0 or "TRITON_EXECUTION_OK" not in env.stdout:
        print("GPU environment is not viable; refusing fused8 model run.")
        return env.returncode or 1

    for config in (args.baseline_config, args.guided_config):
        check = run_command([sys.executable, "-m", "openkernelforge.cli", "check-backend", "--config", config])
        _print_output(check)
        if check.returncode != 0:
            print(f"Backend unavailable for {config}; fused8 model run stopped.")
            print("If this is an OpenAI-compatible local server, check base_url/model/API key requirements.")
            return check.returncode or 1

    baseline = _run_gpu_config(args.baseline_config, f"{args.out_name}_baseline")
    if baseline.returncode != 0 or not baseline.run_dir:
        return baseline.returncode or 1
    guided = _run_gpu_config(args.guided_config, f"{args.out_name}_template_guided")
    if guided.returncode != 0 or not guided.run_dir:
        return guided.returncode or 1

    for run_dir in (baseline.run_dir, guided.run_dir):
        _print_output(run_command([sys.executable, "-m", "openkernelforge.cli", "fused8-report", "--run-dir", str(run_dir)]))
        _print_output(run_command([sys.executable, "-m", "openkernelforge.cli", "repeatability-report", "--run-dir", str(run_dir), "--top-k", "3", "--repeats", "5"]))

    compare = run_command(
        [
            sys.executable,
            "scripts/compare_models_fused8.py",
            "--template",
            args.template_run,
            "--gemini",
            args.gemini_run,
            "--gemini-guided",
            args.gemini_guided_run,
            "--strong",
            str(baseline.run_dir),
            "--strong-guided",
            str(guided.run_dir),
            "--out",
            f"runs/{args.out_name}_comparison.md",
        ]
    )
    _print_output(compare)
    if compare.returncode != 0:
        return compare.returncode
    print("")
    print("Strong model fused8 run complete.")
    print(f"Baseline run: {baseline.run_dir}")
    print(f"Template-guided run: {guided.run_dir}")
    print(f"Comparison: runs/{args.out_name}_comparison.md")
    return 0


class RunResult:
    def __init__(self, returncode: int, run_dir: Path | None):
        self.returncode = returncode
        self.run_dir = run_dir


def _run_gpu_config(config: str, out_name: str) -> RunResult:
    result = run_command(
        [
            sys.executable,
            "scripts/run_gpu_baseline_3tasks.py",
            "--config",
            config,
            "--out-name",
            out_name,
        ]
    )
    _print_output(result)
    return RunResult(result.returncode, _parse_run_dir(result.stdout))


def _parse_run_dir(output: str) -> Path | None:
    match = re.search(r"Run directory:\s*(\S+)", output) or re.search(r"Run complete:\s*(\S+)", output)
    return Path(match.group(1)) if match else None


def _print_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")


if __name__ == "__main__":
    raise SystemExit(main())
