#!/usr/bin/env python3
"""Build the main workshop figure from completed corrected campaign artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--holdout",
        default="artifacts/workshop2026/holdout_campaign/analysis/holdout_confirmation.csv",
    )
    parser.add_argument(
        "--multiplicity",
        default="reports/tables/workshop2026_near_threshold_multiplicity.csv",
    )
    parser.add_argument(
        "--lifecycle",
        default="artifacts/workshop2026/lifecycle_ablation/lifecycle_ablation.csv",
    )
    parser.add_argument(
        "--output",
        default="paper/workshop2026/figures/corrected_campaign_results.pdf",
    )
    args = parser.parse_args()
    paths = [Path(args.holdout), Path(args.multiplicity), Path(args.lifecycle)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "corrected result figure requires completed artifacts: " + ", ".join(missing)
        )
    build_figure(paths[0], paths[1], paths[2], Path(args.output))
    print(f"workshop result figure: {args.output}")
    return 0


def build_figure(
    holdout_path: Path,
    multiplicity_path: Path,
    lifecycle_path: Path,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    holdout = _read_csv(holdout_path)
    multiplicity = _read_csv(multiplicity_path)
    lifecycle = _read_csv(lifecycle_path)
    paired = [
        (float(row["screening_speedup"]), float(row["confirmation_speedup"]))
        for row in holdout
        if row.get("screening_speedup") and row.get("confirmation_speedup")
    ]
    if not paired or not multiplicity or not lifecycle:
        raise ValueError("completed corrected artifacts contain no plottable rows")

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0), constrained_layout=True)
    color = "#2F5D7C"
    accent = "#9B4A3C"

    x_values, y_values = zip(*paired, strict=True)
    axes[0].scatter(x_values, y_values, s=22, color=color, alpha=0.8, edgecolors="none")
    low = min([0.98, *x_values, *y_values])
    high = max([1.02, *x_values, *y_values])
    axes[0].plot([low, high], [low, high], color="0.35", linewidth=1)
    axes[0].axhline(1.02, color=accent, linestyle="--", linewidth=0.9)
    axes[0].axvline(1.02, color=accent, linestyle="--", linewidth=0.9)
    axes[0].set(xlabel="Screening speedup", ylabel="Confirmation speedup", title="A  Holdout")

    budgets = [int(row["candidate_budget"]) for row in multiplicity]
    apparent = [float(row["apparent_win_rate"]) for row in multiplicity]
    confirmed = [float(row["confirmed_win_rate"]) for row in multiplicity]
    axes[1].plot(
        budgets,
        apparent,
        marker="o",
        color=accent,
        linewidth=1.3,
        label="apparent",
    )
    axes[1].plot(
        budgets,
        confirmed,
        marker="s",
        color=color,
        linewidth=1.3,
        label="confirmed",
    )
    axes[1].fill_between(budgets, confirmed, apparent, color="0.75", alpha=0.35)
    axes[1].set(
        xlabel="Candidate budget K",
        ylabel="Win rate above 1.02x",
        title="B  Near-threshold stress test",
        xticks=budgets,
        ylim=(0.0, 0.82),
    )
    axes[1].legend(frameon=False, loc="upper left", fontsize=7)

    inflation = [float(row["median_host_lifecycle_inflation"]) for row in lifecycle]
    axes[2].boxplot(
        inflation,
        orientation="vertical",
        widths=0.4,
        patch_artist=True,
        boxprops={"facecolor": color, "alpha": 0.75},
        medianprops={"color": "white", "linewidth": 1.4},
    )
    axes[2].axhline(1.0, color="0.35", linewidth=1)
    axes[2].set(
        ylabel="Reconstruct/persistent host latency",
        title="C  Lifecycle inflation",
        xticks=[1],
        xticklabels=["control tasks"],
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
