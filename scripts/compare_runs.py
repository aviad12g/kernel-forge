from __future__ import annotations

import argparse

from openkernelforge.reports.compare import compare_runs_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare OpenKernelForge run directories")
    parser.add_argument("run_dirs", nargs="+", help="Run directories to compare")
    args = parser.parse_args()
    print(compare_runs_markdown(args.run_dirs), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
