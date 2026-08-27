#!/usr/bin/env python3
"""Freeze a performance-blind KernelBench L1 task manifest."""

from __future__ import annotations

import argparse

from openkernelforge.tasks.selection_manifest import freeze_kernelbench_selection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/workshop2026_holdout_protocol.yaml",
    )
    parser.add_argument("--kernelbench-dir", required=True)
    parser.add_argument("--output-root")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly replace an existing frozen manifest; never used in the prespecified campaign.",
    )
    args = parser.parse_args()
    paths = freeze_kernelbench_selection(
        args.protocol,
        args.kernelbench_dir,
        output_root=args.output_root,
        replace=args.replace,
    )
    print(f"manifest: {paths.manifest}")
    print(f"csv: {paths.csv}")
    print(f"checksum: {paths.checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
