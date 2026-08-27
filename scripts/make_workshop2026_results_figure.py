#!/usr/bin/env python3
"""Build the main workshop figure from completed corrected campaign artifacts."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
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

    plt.rcParams.update(
        {
            "font.size": 8.4,
            "font.family": "DejaVu Serif",
            "axes.titleweight": "semibold",
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.5, 2.25),
        gridspec_kw={"width_ratios": [1.0, 1.08, 1.0]},
        constrained_layout=True,
    )
    navy = "#244A63"
    rust = "#A4513D"
    gold = "#C19A55"
    charcoal = "#31363B"
    grid = "#D9DEE2"

    x_values, y_values = zip(*paired, strict=True)
    low = max(0.0, min([*x_values, *y_values]) - 0.04)
    high = max(1.06, max([*x_values, *y_values]) + 0.04)
    axes[0].scatter(
        x_values,
        y_values,
        s=30,
        color=navy,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    axes[0].plot([low, high], [low, high], color=charcoal, linewidth=0.9, zorder=1)
    axes[0].axhline(1.02, color=rust, linestyle=(0, (4, 3)), linewidth=1.0)
    axes[0].axvline(1.02, color=rust, linestyle=(0, (4, 3)), linewidth=1.0)
    axes[0].text(
        0.04,
        0.94,
        "0/10 confirmed above margin",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color=charcoal,
    )
    axes[0].set(
        xlabel="Screening speedup vs. eager",
        ylabel="Confirmation speedup vs. eager",
        title="A  Holdout confirmation",
        xlim=(low, high),
        ylim=(low, high),
    )

    budgets = [int(row["candidate_budget"]) for row in multiplicity]
    apparent = [float(row["apparent_win_rate"]) for row in multiplicity]
    confirmed = [float(row["confirmed_win_rate"]) for row in multiplicity]
    axes[1].plot(
        budgets,
        apparent,
        marker="o",
        color=rust,
        linewidth=1.7,
        label="Screening",
    )
    axes[1].plot(
        budgets,
        confirmed,
        marker="s",
        color=navy,
        linewidth=1.7,
        label="Confirmation",
    )
    axes[1].fill_between(budgets, confirmed, apparent, color=gold, alpha=0.18)
    axes[1].text(
        budgets[-1] - 0.12,
        apparent[-1] + 0.026,
        f"Screening {apparent[-1]:.2f}",
        color=rust,
        ha="right",
        va="bottom",
        fontsize=7.5,
    )
    axes[1].text(
        budgets[-1] - 0.12,
        confirmed[-1] - 0.07,
        f"Confirmation {confirmed[-1]:.2f}",
        color=navy,
        ha="right",
        va="top",
        fontsize=7.5,
    )
    axes[1].set(
        xlabel="Candidate budget, K",
        ylabel="Win rate above 1.02x",
        title="B  Parity stress test (4 tasks)",
        xticks=budgets,
        ylim=(0.0, 0.86),
    )

    host_inflation = [float(row["median_host_lifecycle_inflation"]) for row in lifecycle]
    event_inflation = [float(row["median_enclosing_event_inflation"]) for row in lifecycle]
    axes[2].boxplot(
        [host_inflation, event_inflation],
        orientation="vertical",
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": charcoal, "linewidth": 0.9},
        medianprops={"color": "white", "linewidth": 1.4},
        whiskerprops={"color": charcoal, "linewidth": 0.8},
        capprops={"color": charcoal, "linewidth": 0.8},
    )
    for patch, facecolor in zip(axes[2].patches, [rust, navy], strict=True):
        patch.set_facecolor(facecolor)
        patch.set_alpha(0.72)
    rng = random.Random(20260827)
    for index, (values, point_color) in enumerate(
        [(host_inflation, rust), (event_inflation, navy)], start=1
    ):
        jitter = [index + rng.uniform(-0.13, 0.13) for _ in values]
        axes[2].scatter(
            jitter,
            values,
            s=10,
            color=point_color,
            alpha=0.45,
            edgecolors="none",
            zorder=3,
        )
        median = statistics.median(values)
        axes[2].text(
            index,
            median + 0.08,
            f"{median:.3f}x",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="semibold",
            color=charcoal,
        )
    axes[2].axhline(1.0, color=charcoal, linewidth=0.9)
    axes[2].set(
        ylabel="Reconstruct / persistent ratio",
        title="C  Timing boundary (24 rows)",
        xticks=[1, 2],
        xticklabels=["Host\nend to end", "CUDA\nevent"],
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color=grid, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(length=3, width=0.7, color=charcoal)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": "OpenKernelForge corrected campaign results"},
    )
    plt.close(fig)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
