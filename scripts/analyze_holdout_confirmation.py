#!/usr/bin/env python3
"""Select screening winners and analyze fresh-process confirmation records."""

from __future__ import annotations

import argparse
from pathlib import Path

from openkernelforge.reports.holdout_confirmation import (
    analyze_holdout_confirmation,
    read_timing_blocks,
    select_screening_winners,
    write_promotion_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, help="CSV containing screening and confirmation blocks")
    parser.add_argument("--output-dir", default="reports/workshop2026")
    parser.add_argument("--practical-margin", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=62_081)
    parser.add_argument("--fdr", type=float, default=0.05)
    args = parser.parse_args()

    records = read_timing_blocks(args.records)
    winners = select_screening_winners(records)
    results = analyze_holdout_confirmation(
        records,
        winners,
        practical_margin=args.practical_margin,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        false_discovery_rate=args.fdr,
    )
    paths = write_promotion_artifacts(Path(args.output_dir), winners, results)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
