from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CONFIG = Path("configs/real_baseline_3tasks.yaml")


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def main() -> int:
    check = run_command(
        [sys.executable, "-m", "openkernelforge.cli", "check-backend", "--config", str(CONFIG)]
    )
    if check.returncode != 0:
        print("Backend unavailable; real baseline was not run.")
        _print_output(check)
        print("Start an OpenAI-compatible server, then retry:")
        print("  python -m vllm.entrypoints.openai.api_server --model <model> --host 0.0.0.0 --port 8000")
        print(f"  {sys.executable} -m openkernelforge.cli check-backend --config {CONFIG}")
        return check.returncode or 1

    run = run_command([sys.executable, "-m", "openkernelforge.cli", "run", "--config", str(CONFIG)])
    _print_output(run)
    if run.returncode != 0:
        print("Real baseline run failed.")
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

    dataset_dir = Path("datasets") / f"real_baseline_3tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    print("")
    print("Real baseline complete.")
    print(f"Run directory: {run_dir}")
    print(f"Analysis path: {run_dir / 'analysis.md'}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Failure taxonomy counts: {manifest.get('counts_by_failure_type', {})}")
    print(f"Dataset counts: {manifest.get('counts_by_file', {})}")
    return 0


def _parse_run_dir(output: str) -> Path | None:
    match = re.search(r"Run complete:\s*(\S+)", output)
    return Path(match.group(1)) if match else None


def _load_manifest(dataset_dir: Path) -> dict:
    path = dataset_dir / "manifest.json"
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
