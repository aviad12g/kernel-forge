#!/usr/bin/env python3
"""Summarize uncertainty in the preserved lifecycle-ablation process rows."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    ("host_lifecycle_inflation", "median_host_lifecycle_inflation"),
    ("enclosing_event_inflation", "median_enclosing_event_inflation"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="artifacts/workshop2026/lifecycle_ablation/lifecycle_ablation.csv",
    )
    parser.add_argument(
        "--csv-output",
        default="reports/tables/workshop2026_lifecycle_uncertainty.csv",
    )
    parser.add_argument(
        "--report-output",
        default="reports/workshop2026_lifecycle_uncertainty.md",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_827)
    args = parser.parse_args()

    input_path = _resolve(args.input)
    summaries = summarize_lifecycle_rows(
        input_path,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    csv_output = _resolve(args.csv_output)
    report_output = _resolve(args.report_output)
    _write_csv(csv_output, summaries)
    _write_report(report_output, summaries, input_path)
    print(f"lifecycle uncertainty table: {csv_output}")
    print(f"lifecycle uncertainty report: {report_output}")
    return 0


def summarize_lifecycle_rows(
    path: Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("lifecycle CSV contains no process rows")

    task_ids = sorted({str(row["task_id"]) for row in rows})
    summaries: list[dict[str, object]] = []
    for metric_index, (metric, field) in enumerate(METRICS):
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped[str(row["task_id"])].append(float(row[field]))
        values = [value for task_id in task_ids for value in grouped[task_id]]
        rng = random.Random(seed + metric_index)
        bootstrap_medians: list[float] = []
        for _ in range(bootstrap_samples):
            sample: list[float] = []
            for _task in task_ids:
                sample.extend(grouped[rng.choice(task_ids)])
            bootstrap_medians.append(statistics.median(sample))
        q25 = _percentile(values, 0.25)
        q75 = _percentile(values, 0.75)
        summaries.append(
            {
                "metric": metric,
                "tasks": len(task_ids),
                "process_rows": len(values),
                "median": statistics.median(values),
                "q25": q25,
                "q75": q75,
                "iqr": q75 - q25,
                "task_cluster_bootstrap_lo": _percentile(bootstrap_medians, 0.025),
                "task_cluster_bootstrap_hi": _percentile(bootstrap_medians, 0.975),
                "bootstrap_samples": bootstrap_samples,
                "seed": seed,
                "artifact_source": path.relative_to(ROOT).as_posix()
                if path.is_relative_to(ROOT)
                else str(path),
                "notes": (
                    "IQR is over process-level medians; bootstrap resamples tasks and "
                    "retains all three process rows for each sampled task"
                ),
            }
        )
    return summaries


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, object]], source: Path) -> None:
    source_label = source.relative_to(ROOT) if source.is_relative_to(ROOT) else source
    lines = [
        "# Workshop 2026 Lifecycle-Ablation Uncertainty",
        "",
        "This report uses only the 24 preserved process rows from the completed control. "
        "It does not rerun CUDA work.",
        "",
        f"Source: `{source_label}`",
        "",
        "| Metric | Median | Process-row IQR | Task-cluster bootstrap 95% interval |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {str(row['metric']).replace('_', ' ')} | {float(row['median']):.6f} | "
            f"[{float(row['q25']):.6f}, {float(row['q75']):.6f}] | "
            f"[{float(row['task_cluster_bootstrap_lo']):.6f}, "
            f"{float(row['task_cluster_bootstrap_hi']):.6f}] |"
        )
    lines.extend(
        [
            "",
            "The IQR describes dispersion across process-level medians. The bootstrap "
            "resamples the eight tasks as clusters and retains the three process rows "
            "associated with each sampled task. The broad cluster intervals reflect "
            "task heterogeneity and are not evidence of a population-wide effect.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
