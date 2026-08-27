#!/usr/bin/env python3
"""Compare the original and same-GPU controlled multiplicity summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original",
        default="artifacts/workshop2026/multiplicity/campaign/selection_multiplicity.csv",
    )
    parser.add_argument(
        "--same-gpu",
        default="artifacts/workshop2026/multiplicity_same_gpu_a4500/selection_multiplicity.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="reports/tables/workshop2026_same_gpu_rq2_control.csv",
    )
    parser.add_argument(
        "--output-report",
        default="reports/workshop2026_same_gpu_rq2_control.md",
    )
    args = parser.parse_args()

    original = _read_rows(_resolve(args.original))
    same_gpu = _read_rows(_resolve(args.same_gpu))
    rows = compare_rows(original, same_gpu)
    output_csv = _resolve(args.output_csv)
    output_report = _resolve(args.output_report)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_csv, rows)
    _write_report(output_report, rows)
    print(f"same-GPU RQ2 table: {output_csv}")
    print(f"same-GPU RQ2 report: {output_report}")
    return 0


def compare_rows(
    original: list[dict[str, str]], same_gpu: list[dict[str, str]]
) -> list[dict[str, Any]]:
    by_budget = {int(row["candidate_budget"]): row for row in same_gpu}
    rows: list[dict[str, Any]] = []
    for old in original:
        budget = int(old["candidate_budget"])
        if budget not in by_budget:
            raise RuntimeError(f"same-GPU summary is missing candidate budget {budget}")
        new = by_budget[budget]
        rows.append(
            {
                "candidate_budget": budget,
                "original_gpu": "T4",
                "original_apparent_win_rate": float(old["apparent_win_rate"]),
                "original_confirmed_win_rate": float(old["confirmed_win_rate"]),
                "same_gpu": "RTX A4500",
                "same_gpu_apparent_win_rate": float(new["apparent_win_rate"]),
                "same_gpu_confirmed_win_rate": float(new["confirmed_win_rate"]),
                "same_gpu_median_selection_optimism_log": float(
                    new["median_selection_optimism_log"]
                ),
                "same_gpu_eligible_tasks": int(new["eligible_tasks"]),
                "interpretation": _interpret(new),
            }
        )
    return rows


def _interpret(row: dict[str, str]) -> str:
    apparent = float(row["apparent_win_rate"])
    confirmed = float(row["confirmed_win_rate"])
    if apparent == confirmed:
        return "no_screen_confirmation_gap"
    if apparent > confirmed:
        return "screening_exceeds_confirmation"
    return "confirmation_exceeds_screening"


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty multiplicity summary: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Same-GPU RQ2 Control",
        "",
        "This control reruns the frozen easy deterministic candidate grid on the RTX A4500 used for the near-threshold study. It removes the earlier cross-GPU comparison without changing either primary result.",
        "",
        "| K | Original T4 apparent | Original T4 confirmed | A4500 apparent | A4500 confirmed | A4500 median optimism (log) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate_budget']} | "
            f"{row['original_apparent_win_rate']:.4f} | "
            f"{row['original_confirmed_win_rate']:.4f} | "
            f"{row['same_gpu_apparent_win_rate']:.4f} | "
            f"{row['same_gpu_confirmed_win_rate']:.4f} | "
            f"{row['same_gpu_median_selection_optimism_log']:.6f} |"
        )
    kmax = max(rows, key=lambda row: int(row["candidate_budget"]))
    lines.extend(
        [
            "",
            f"At K={kmax['candidate_budget']}, the A4500 apparent and confirmed win rates are {kmax['same_gpu_apparent_win_rate']:.4f} and {kmax['same_gpu_confirmed_win_rate']:.4f}. This is a bounded hardware-control result for the four-task easy grid, not a broader claim about model-generated candidates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
