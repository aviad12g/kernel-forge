from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_FIGURES = ROOT / "reports" / "figures"
OVERLEAF_FIGURES = ROOT / "paper" / "overleaf" / "figures"


PALETTE = {
    "template": "#4C78A8",
    "gemini": "#59A14F",
    "openai": "#F28E2B",
    "neutral": "#6B7280",
    "warning": "#B79A20",
    "danger": "#D65F5F",
    "line": "#111827",
    "grid": "#E5E7EB",
}


def main() -> int:
    make_figures()
    return 0


def make_figures() -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except Exception as exc:  # pragma: no cover - dependency check
        raise SystemExit(f"matplotlib is required to build paper figures: {exc}") from exc

    for directory in (REPORT_FIGURES, OVERLEAF_FIGURES):
        directory.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )

    paths: list[Path] = []
    paths.extend(_pipeline(plt, FancyBboxPatch))
    paths.extend(_fused8_stable(plt))
    paths.extend(_fused8_sources(plt))
    paths.extend(_bias_relu(plt))
    paths.extend(_kernelbench_funnel(plt))
    paths.extend(_failure_taxonomy(plt))
    return paths


def _save(fig, name: str) -> list[Path]:
    paths: list[Path] = []
    for directory in (REPORT_FIGURES, OVERLEAF_FIGURES):
        png = directory / f"{name}.png"
        pdf = directory / f"{name}.pdf"
        fig.savefig(png, dpi=220)
        fig.savefig(pdf)
        paths.extend([png, pdf])
    return paths


def _pipeline(plt, FancyBboxPatch) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10.5, 2.35))
    ax.set_axis_off()
    labels = [
        "Prompt or\ntemplate",
        "Candidate\nsource",
        "Policy\ncheck",
        "Correctness\nverification",
        "CUDA-event\nbenchmark",
        "Repeatability\nlabel",
        "Report and\ndataset",
    ]
    colors = ["#F7F7F7", "#F7F7F7", "#EEF2FF", "#ECFDF5", "#FFF7ED", "#FDF2F8", "#F7F7F7"]
    width = 0.115
    y = 0.42
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = 0.02 + i * 0.14
        box = FancyBboxPatch(
            (x, y),
            width,
            0.34,
            boxstyle="round,pad=0.014,rounding_size=0.018",
            linewidth=1.0,
            edgecolor="#4B5563",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + width / 2, y + 0.17, label, ha="center", va="center", fontsize=9.6, weight="semibold")
        if i < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x + width + 0.02, y + 0.17),
                xytext=(x + width + 0.004, y + 0.17),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.2),
            )
    ax.text(0.5, 0.18, "Each candidate keeps prompt, response, source, verification, timing samples, and final label.", ha="center", fontsize=9.4, color="#374151")
    return _save(fig, "openkernelforge_pipeline")


def _fused8_stable(plt) -> list[Path]:
    rows = [
        ("residual", "OpenAI mini", 1.074, PALETTE["openai"]),
        ("bias_gelu", "template", 1.485, PALETTE["template"]),
        ("rmsnorm", "template", 1.452, PALETTE["template"]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar([r[0] for r in rows], [r[2] for r in rows], color=[r[3] for r in rows], width=0.62)
    ax.axhline(1.0, color=PALETTE["line"], lw=1.0)
    ax.set_ylim(0.85, 1.62)
    ax.set_ylabel("repeat median speedup vs eager")
    ax.set_title("Repeat-stable fused8 winners")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, (_, source, value, _) in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}x\n{source}", ha="center", va="bottom", fontsize=9.5)
    return _save(fig, "fused8_stable_speedups")


def _fused8_sources(plt) -> list[Path]:
    sources = ["template", "Gemini", "OpenAI mini"]
    candidates = [160, 24, 24]
    verified = [100.0, 95.8, 50.0]
    wins = [3, 2, 1]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.8))
    specs = [
        ("Candidates", candidates, "#6B7280", None),
        ("Verification rate", verified, "#59A14F", "%"),
        ("Stable wins", wins, "#4C78A8", None),
    ]
    for ax, (title, values, color, suffix) in zip(axes, specs):
        bars = ax.bar(sources, values, color=color, width=0.58)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
        ymax = max(values) * 1.22 if max(values) else 1
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="x", labelrotation=18)
        for bar, value in zip(bars, values):
            label = f"{value:.0f}%" if suffix == "%" else str(int(value))
            ax.text(bar.get_x() + bar.get_width() / 2, value + ymax * 0.03, label, ha="center", va="bottom", fontsize=9)
    fig.suptitle("Fused8 source comparison", y=1.03, fontsize=12, weight="semibold")
    fig.tight_layout()
    return _save(fig, "fused8_source_summary")


def _bias_relu(plt) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    values = [1.029, 0.976]
    labels = ["single run", "repeat median"]
    colors = ["#9CA3AF", PALETTE["danger"]]
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.axhline(1.0, color=PALETTE["line"], lw=1.0)
    ax.set_ylim(0.90, 1.06)
    ax.set_ylabel("speedup vs eager")
    ax.set_title("bias_relu: repeatability changes the label")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}x", ha="center", va="bottom", fontsize=9.5)
    ax.text(1, 0.914, "SINGLE_RUN_ONLY_WIN", ha="center", va="center", fontsize=8.8, color="#991B1B", weight="bold")
    return _save(fig, "bias_relu_single_run_flip")


def _kernelbench_funnel(plt) -> list[Path]:
    labels = ["selected\ntasks", "generated\ncandidates", "one-shot\nverified", "one-shot\nstable", "repairs\nattempted", "repair\nverified", "total\nstable"]
    values = [20, 20, 3, 2, 8, 1, 3]
    colors = [PALETTE["neutral"], PALETTE["neutral"], PALETTE["gemini"], PALETTE["gemini"], PALETTE["warning"], PALETTE["warning"], PALETTE["gemini"]]
    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    bars = ax.bar(labels, values, color=colors, width=0.68)
    ax.set_ylabel("count")
    ax.set_title("Historical KernelBench adapter output funnel")
    ax.set_ylim(0, 22.5)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.55, str(value), ha="center", va="bottom", fontsize=9.5)
    return _save(fig, "kernelbench_pilot_funnel")


def _failure_taxonomy(plt) -> list[Path]:
    labels = ["numerical\nmismatch", "timeout\n/OOM", "runtime", "Triton\ncompile"]
    values = [9, 4, 2, 2]
    colors = [PALETTE["danger"], PALETTE["warning"], PALETTE["neutral"], "#8B5CF6"]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylabel("failed candidates")
    ax.set_title("Historical KernelBench verifier taxonomy")
    ax.set_ylim(0, 10.5)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, str(value), ha="center", va="bottom", fontsize=9.5)
    return _save(fig, "kernelbench_failure_taxonomy")


if __name__ == "__main__":
    raise SystemExit(main())
