#!/usr/bin/env python3
"""Analyze candidate-budget optimism from preserved independent timings."""

from __future__ import annotations

import argparse

from openkernelforge.reports.holdout_confirmation import read_timing_blocks
from openkernelforge.reports.selection_multiplicity import (
    analyze_selection_multiplicity,
    write_multiplicity_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-blocks", required=True)
    parser.add_argument(
        "--output",
        default="reports/tables/selection_multiplicity.csv",
    )
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    records = read_timing_blocks(args.timing_blocks)
    rows = analyze_selection_multiplicity(
        records,
        resamples=args.resamples,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = write_multiplicity_csv(args.output, rows)
    print(f"selection multiplicity: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
