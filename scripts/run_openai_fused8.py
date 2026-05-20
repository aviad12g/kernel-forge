from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OpenAI fused8 baseline and template-guided comparison.")
    parser.add_argument("--baseline-config", default="configs/openai_fused8_gpu_baseline.yaml")
    parser.add_argument("--guided-config", default="configs/openai_fused8_gpu_template_guided.yaml")
    parser.add_argument("--out-name", default="openai_gpt55_fused8")
    args = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.")
        print("export OPENAI_API_KEY=<your-key>")
        return 1

    command = [
        sys.executable,
        "scripts/run_strong_model_fused8.py",
        "--baseline-config",
        args.baseline_config,
        "--guided-config",
        args.guided_config,
        "--out-name",
        args.out_name,
    ]
    result = run_command(command)
    _print_output(result)
    if result.returncode != 0:
        print("")
        print("OpenAI fused8 run failed.")
        print("Check whether this was authentication, model availability, or endpoint/API-mode related.")
        print(
            "If chat-completions is not supported for the selected model, try "
            "configs/openai_responses_fused8_gpu_baseline.yaml and "
            "configs/openai_responses_fused8_gpu_template_guided.yaml."
        )
        return result.returncode
    print("OpenAI fused8 run complete. Consider running: unset OPENAI_API_KEY")
    return 0


def _print_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")


if __name__ == "__main__":
    raise SystemExit(main())
