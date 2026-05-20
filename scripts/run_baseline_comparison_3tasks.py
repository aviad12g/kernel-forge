from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


DUMMY_CONFIG = Path("configs/dummy_baseline_3tasks.yaml")
FAKE_CONFIG = Path("configs/fake_baseline_3tasks.yaml")
REAL_CONFIG = Path("configs/real_baseline_3tasks.yaml")


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def main() -> int:
    run_dirs: list[Path] = []

    for label, config in [("dummy", DUMMY_CONFIG), ("fake", FAKE_CONFIG)]:
        result = run_command([sys.executable, "-m", "openkernelforge.cli", "run", "--config", str(config)])
        _print_output(result)
        if result.returncode != 0:
            print(f"{label} baseline failed.")
            return result.returncode
        run_dir = _parse_run_dir(result.stdout)
        if run_dir is None:
            print(f"Could not determine {label} run directory.")
            return 1
        run_dirs.append(run_dir)

    check = run_command(
        [sys.executable, "-m", "openkernelforge.cli", "check-backend", "--config", str(REAL_CONFIG)]
    )
    if check.returncode == 0:
        real = run_command([sys.executable, "-m", "openkernelforge.cli", "run", "--config", str(REAL_CONFIG)])
        _print_output(real)
        if real.returncode != 0:
            print("Real baseline failed.")
            return real.returncode
        real_dir = _parse_run_dir(real.stdout)
        if real_dir is not None:
            run_dirs.append(real_dir)
    else:
        print("Real backend unavailable; skipped real baseline.")
        _print_output(check)

    compare = run_command(
        [sys.executable, "scripts/compare_runs.py", *[str(run_dir) for run_dir in run_dirs]]
    )
    _print_output(compare)
    return compare.returncode


def _parse_run_dir(output: str) -> Path | None:
    match = re.search(r"Run complete:\s*(\S+)", output)
    return Path(match.group(1)) if match else None


def _print_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")


if __name__ == "__main__":
    raise SystemExit(main())
