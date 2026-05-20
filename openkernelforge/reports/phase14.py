"""Research report and reproducibility artifact generation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TABLE_NAMES = [
    "fused8_model_comparison.csv",
    "fused8_template_results.csv",
    "fused8_stable_winners.csv",
    "curated_dataset_counts.csv",
    "three_task_summary.csv",
]


@dataclass(frozen=True)
class Artifact:
    label: str
    canonical_path: str
    local_path: str
    description: str


THREE_TASK_ROWS = [
    {
        "task": "vector_add",
        "best_single_run_speedup_vs_eager": "0.692",
        "repeatability_outcome": "repeat median 0.483x",
        "final_conclusion": "poor standalone target; launch overhead dominates",
    },
    {
        "task": "relu",
        "best_single_run_speedup_vs_eager": "0.812",
        "repeatability_outcome": "repeat median 0.512x",
        "final_conclusion": "poor standalone target; use only as harness check",
    },
    {
        "task": "bias_relu",
        "best_single_run_speedup_vs_eager": "1.017",
        "repeatability_outcome": "repeat median 0.705x",
        "final_conclusion": "single-run win was not stable; useful as fused-task seed",
    },
]

FUSED8_TEMPLATE_ROWS = [
    ("bias_relu", "1.017", "0.954", "no", "yes"),
    ("sigmoid_mul", "1.065", "0.865", "no", "yes"),
    ("add_relu", "0.924", "not above eager", "no", "yes"),
    ("residual_add_relu", "1.378", "1.168", "yes", "yes"),
    ("bias_gelu", "1.697", "1.657", "yes", "yes"),
    ("row_sum", "0.801", "not above eager", "no", "yes"),
    ("layernorm_small", "0.843", "not above eager", "no", "yes"),
    ("rmsnorm_small", "2.227", "1.802", "yes", "yes"),
]

MODEL_COMPARISON_ROWS = [
    {
        "baseline": "deterministic templates",
        "candidates": "2076",
        "verified": "2076/2076",
        "median_speedup_vs_eager": "0.862",
        "tasks_above_eager": "5/8 single-run",
        "repeat_stable_wins": "residual_add_relu, bias_gelu, rmsnorm_small",
        "conclusion": "strongest overall floor; repeatability required",
    },
    {
        "baseline": "Gemini fused8 baseline",
        "candidates": "28",
        "verified": "28/28",
        "median_speedup_vs_eager": "0.933",
        "tasks_above_eager": "4/8 single-run",
        "repeat_stable_wins": "competitive but not final stable winner in provided summary",
        "conclusion": "strong zero-shot correctness and competitive speed",
    },
    {
        "baseline": "Gemini template-guided",
        "candidates": "34",
        "verified": "34/34",
        "median_speedup_vs_eager": "0.798",
        "tasks_above_eager": "4/8 single-run",
        "repeat_stable_wins": "residual_add_relu",
        "conclusion": "useful optimization data; median performance regressed",
    },
    {
        "baseline": "OpenAI mini cheap",
        "candidates": "8",
        "verified": "8/8",
        "median_speedup_vs_eager": "0.882",
        "tasks_above_eager": "3/8 single-run",
        "repeat_stable_wins": "residual_add_relu, bias_gelu, rmsnorm_small",
        "conclusion": "cheap and competitive; not clearly above Gemini",
    },
    {
        "baseline": "GPT-5.5 cheap",
        "candidates": "8",
        "verified": "8/8",
        "median_speedup_vs_eager": "0.927",
        "tasks_above_eager": "provided summary: not dominant",
        "repeat_stable_wins": "bias_gelu, rmsnorm_small",
        "conclusion": "correct and usable; not clearly better under cheap protocol",
    },
    {
        "baseline": "Qwen 7B local",
        "candidates": "8",
        "verified": "1/8 effective",
        "median_speedup_vs_eager": "0.002",
        "tasks_above_eager": "0/8",
        "repeat_stable_wins": "none",
        "conclusion": "not competitive zero-shot",
    },
]

CURATED_DATASET_ROWS = [
    ("correct_fast_repeat_stable.jsonl", "19", "reviewed SFT candidates after repeatability", "not yet"),
    ("correct_fast_single_run.jsonl", "623", "analysis and candidate mining only", "no"),
    ("correct_promising.jsonl", "598", "optimization or ranking data after review", "no"),
    ("optimization_pairs_template_vs_gemini.jsonl", "5", "optimization training pairs", "not yet"),
    ("optimization_pairs_gemini_vs_template.jsonl", "3", "optimization training pairs", "not yet"),
    ("rejected_or_unstable.jsonl", "898", "failure analysis and negative examples", "no"),
]

STABLE_WINNER_ROWS = [
    ("bias_relu", "none confirmed above eager", "n/a", "n/a", "single-run wins were not repeat-stable"),
    ("sigmoid_mul", "none confirmed above eager", "n/a", "n/a", "single-run win did not hold above eager"),
    ("add_relu", "none confirmed above eager", "n/a", "n/a", "near-eager only"),
    ("residual_add_relu", "Gemini template-guided", "llm_template_guided", "1.234", "LLM-guided run produced the stable winner"),
    ("bias_gelu", "deterministic template", "template", "1.657", "strong repeat-stable deterministic template win"),
    ("row_sum", "none confirmed above eager", "n/a", "n/a", "below eager in current protocol"),
    ("layernorm_small", "none confirmed above eager", "n/a", "n/a", "below eager in current protocol"),
    ("rmsnorm_small", "deterministic template", "template", "1.802", "strong repeat-stable deterministic template win"),
]

ARTIFACTS = [
    Artifact(
        "deterministic fused8 template wide",
        "/workspace/openkernelforge/runs/20260519_213349",
        "runs/20260519_213349",
        "2076-candidate deterministic template floor",
    ),
    Artifact(
        "Gemini fused8 baseline",
        "/workspace/openkernelforge/runs/20260519_215314",
        "runs/20260519_215314",
        "28-candidate Gemini baseline",
    ),
    Artifact(
        "Gemini fused8 template-guided",
        "/workspace/openkernelforge/runs/20260519_215439",
        "runs/20260519_215439",
        "34-candidate template-guided Gemini run",
    ),
    Artifact(
        "OpenAI mini cheap",
        "/workspace/openkernelforge/runs/20260520_083300",
        "runs/20260520_083300",
        "8-candidate OpenAI mini smoke baseline",
    ),
    Artifact(
        "GPT-5.5 cheap",
        "/workspace/openkernelforge/runs/20260520_085334",
        "runs/20260520_085334",
        "8-candidate GPT-5.5 smoke baseline",
    ),
    Artifact(
        "Qwen 7B local",
        "/workspace/openkernelforge/runs/20260520_114551",
        "runs/20260520_114551",
        "8-candidate local Qwen 7B baseline",
    ),
    Artifact(
        "curated fused8 dataset",
        "/workspace/openkernelforge/datasets/fused8_curated_v1",
        "datasets/fused8_curated_v1",
        "repeatability-aware curated fused8 dataset",
    ),
    Artifact(
        "final fused8 conclusion",
        "/workspace/openkernelforge/runs/fused8_phase11_conclusion.md",
        "runs/fused8_phase11_conclusion.md",
        "final fused8 conclusion report",
    ),
]


def build_phase14_report(root: str | Path = ".") -> list[Path]:
    """Generate research Markdown reports and CSV source tables."""

    root_path = Path(root)
    reports = root_path / "reports"
    tables = reports / "tables"
    figures = reports / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    written.extend(_write_tables(tables))
    written.append(_write_text(reports / "openkernelforge_technical_report.md", _technical_report()))
    written.append(_write_text(reports / "reproducibility.md", _reproducibility_guide()))
    written.append(_write_text(reports / "artifact_index.md", _artifact_index(root_path)))
    _maybe_write_figures(figures)
    return written


def check_research_artifacts(root: str | Path = ".") -> tuple[bool, list[str], list[str]]:
    """Validate that research artifacts exist and tables parse."""

    root_path = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        root_path / "reports/openkernelforge_technical_report.md",
        root_path / "reports/reproducibility.md",
        root_path / "reports/artifact_index.md",
        root_path / "README.md",
    ]
    required.extend(root_path / "reports/tables" / name for name in TABLE_NAMES)
    for path in required:
        if not path.exists():
            errors.append(f"missing required artifact: {path}")

    for name in TABLE_NAMES:
        path = root_path / "reports/tables" / name
        if not path.exists():
            continue
        try:
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except csv.Error as exc:
            errors.append(f"invalid CSV {path}: {exc}")
            continue
        if not rows:
            errors.append(f"empty CSV: {path}")

    readme = root_path / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8", errors="replace").lower()
        if "no sota claim" not in text and "not a sota claim" not in text and "no sota claims" not in text:
            errors.append("README must explicitly state that this is not a SOTA claim")

    curated = root_path / "datasets/fused8_curated_v1"
    if curated.exists() and not (curated / "manifest.json").exists():
        errors.append("curated fused8 dataset exists but manifest.json is missing")
    if not curated.exists():
        warnings.append("datasets/fused8_curated_v1 is not present in this workspace")

    return not errors, errors, warnings


def check_artifacts_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check OpenKernelForge research artifacts.")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    ok, errors, warnings = check_research_artifacts(args.root)
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    if errors:
        print("Artifact check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Artifact check passed.")
    return 0


def build_report_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build OpenKernelForge reports and CSV tables.")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    written = build_phase14_report(args.root)
    print("Generated research artifacts:")
    for path in written:
        print(f"- {path}")
    return 0


def _write_tables(tables: Path) -> list[Path]:
    written = [
        _write_csv(
            tables / "three_task_summary.csv",
            ["task", "best_single_run_speedup_vs_eager", "repeatability_outcome", "final_conclusion"],
            THREE_TASK_ROWS,
        ),
        _write_csv(
            tables / "fused8_template_results.csv",
            [
                "task",
                "best_single_run_speedup_vs_eager",
                "repeat_median_speedup_vs_eager",
                "above_eager_stable",
                "above_torch_compile",
            ],
            [
                {
                    "task": task,
                    "best_single_run_speedup_vs_eager": speedup,
                    "repeat_median_speedup_vs_eager": repeat,
                    "above_eager_stable": stable,
                    "above_torch_compile": compile_win,
                }
                for task, speedup, repeat, stable, compile_win in FUSED8_TEMPLATE_ROWS
            ],
        ),
        _write_csv(
            tables / "fused8_model_comparison.csv",
            [
                "baseline",
                "candidates",
                "verified",
                "median_speedup_vs_eager",
                "tasks_above_eager",
                "repeat_stable_wins",
                "conclusion",
            ],
            MODEL_COMPARISON_ROWS,
        ),
        _write_csv(
            tables / "curated_dataset_counts.csv",
            ["split", "rows", "intended_use", "train_now"],
            [
                {
                    "split": split,
                    "rows": rows,
                    "intended_use": use,
                    "train_now": train_now,
                }
                for split, rows, use, train_now in CURATED_DATASET_ROWS
            ],
        ),
        _write_csv(
            tables / "fused8_stable_winners.csv",
            ["task", "stable_winner", "source_type", "repeat_median_speedup_vs_eager", "interpretation"],
            [
                {
                    "task": task,
                    "stable_winner": winner,
                    "source_type": source,
                    "repeat_median_speedup_vs_eager": repeat,
                    "interpretation": interpretation,
                }
                for task, winner, source, repeat, interpretation in STABLE_WINNER_ROWS
            ],
        ),
    ]
    return written


def _technical_report() -> str:
    return "\n".join(
        [
            "# OpenKernelForge: Verifier-Guided Triton Kernel Generation with Template Baselines and Repeatability-Aware Evaluation",
            "",
            "## 1. Abstract",
            "",
            "OpenKernelForge is an open-source research harness for verifier-guided Triton kernel generation. The current system combines task definitions, model and template candidate generation, static policy checks, correctness verification, benchmarking, repeatability analysis, and dataset export. This report summarizes the first internal fused8 evaluation campaign. The results are intentionally modest: LLMs can generate correct fused Triton kernels, but deterministic templates remain a strong performance floor and repeatability changes the interpretation of many apparent wins. These are internal fused8 results only, not KernelBench results and not a SOTA claim.",
            "",
            "## 2. Motivation",
            "",
            "Kernel generation agents need more than pass/fail execution. Useful systems need reproducible prompts, raw responses, candidate sources, policy checks, correctness traces, timing data, repeatability, and structured failure labels. OpenKernelForge was built to make those artifacts first-class so later open-model fine-tuning can be grounded in verified evidence rather than isolated examples.",
            "",
            "## 3. System Overview",
            "",
            "- Task layer: PyTorch references, deterministic inputs, shape metadata, tolerances, and prompt hints.",
            "- Agent layer: dummy, fake, OpenAI-compatible, local vLLM-compatible, and deterministic template agents.",
            "- Harness layer: candidate extraction, static policy checks, sandboxed import, verifier, benchmarker, and JSONL logging.",
            "- Reporting layer: summaries, run analysis, failure taxonomy, repeatability reports, fused8 reports, and dataset curation.",
            "- Artifact layer: prompt files, raw responses, candidate source, logs, environment probes, datasets, and human-readable reports.",
            "",
            "## 4. Benchmark Tasks",
            "",
            "The project started with a three-task sandbox: `vector_add`, `relu`, and `bias_relu`. That sandbox validated the harness but showed that isolated elementwise tasks are poor standalone performance targets. The project then moved to an internal fused8 benchmark: `bias_relu`, `sigmoid_mul`, `add_relu`, `residual_add_relu`, `bias_gelu`, `row_sum`, `layernorm_small`, and `rmsnorm_small`. These fused workloads are better aligned with Triton launch amortization and realistic kernel-generation behavior.",
            "",
            "## 5. Methods",
            "",
            "Each candidate must expose `forward(*args)`. Candidates pass through policy checks before verification. Correct candidates are benchmarked against PyTorch eager and, when configured, `torch.compile`. Repeatability is measured by rebenchmarking top candidates multiple times. Dataset export separates repeat-stable fast candidates, single-run-only candidates, promising candidates, optimization pairs, and rejected or unstable candidates.",
            "",
            "## 6. Model and Template Baselines",
            "",
            "- Deterministic templates sweep block size, warps, stages, allocation policy, contiguity policy, and shape specialization.",
            "- Gemini fused8 baseline used the cheap fused8 protocol and produced correct kernels reliably.",
            "- Gemini template-guided used deterministic template context but did not improve median speed.",
            "- OpenAI mini and GPT-5.5 cheap runs were correct and competitive but did not clearly dominate Gemini or templates.",
            "- Qwen 7B local zero-shot was weak under the cheap fused8 protocol.",
            "- Qwen 14B was not tested because the vLLM pod ran out of disk/cache space during model download.",
            "",
            "## 7. Results",
            "",
            "Provenance note: the latest RunPod fused8 artifacts are not present in this local checkout, so these tables use the manually provided run summaries for missing artifacts. `reports/artifact_index.md` records which canonical artifacts are absent locally.",
            "",
            "### Three-Task Conclusion",
            "",
            _markdown_table(
                ["Task", "Best single-run speedup", "Repeatability outcome", "Final conclusion"],
                [
                    [row["task"], row["best_single_run_speedup_vs_eager"], row["repeatability_outcome"], row["final_conclusion"]]
                    for row in THREE_TASK_ROWS
                ],
            ),
            "",
            "### Fused8 Deterministic Template Results",
            "",
            _markdown_table(
                ["Task", "Best single-run", "Repeat median", "Stable above eager", "Above torch.compile"],
                [[*row] for row in FUSED8_TEMPLATE_ROWS],
            ),
            "",
            "### Model Comparison",
            "",
            _markdown_table(
                [
                    "Baseline",
                    "Candidates",
                    "Verified",
                    "Median eager speedup",
                    "Tasks above eager",
                    "Repeat-stable wins",
                    "Conclusion",
                ],
                [
                    [
                        row["baseline"],
                        row["candidates"],
                        row["verified"],
                        row["median_speedup_vs_eager"],
                        row["tasks_above_eager"],
                        row["repeat_stable_wins"],
                        row["conclusion"],
                    ]
                    for row in MODEL_COMPARISON_ROWS
                ],
            ),
            "",
            "## 8. Repeatability Analysis",
            "",
            "Repeatability changed several conclusions. In the three-task sandbox, single-run wins for `bias_relu` did not survive repeat benchmarking. In fused8, deterministic templates produced repeat-stable wins for `residual_add_relu`, `bias_gelu`, and `rmsnorm_small`; the final stable winner for `residual_add_relu` came from Gemini template-guided. Single-run wins remain useful search signals, but they are not sufficient evidence for benchmark claims.",
            "",
            "### Stable Winners By Task",
            "",
            _markdown_table(
                ["Task", "Stable winner", "Source type", "Repeat median", "Interpretation"],
                [[*row] for row in STABLE_WINNER_ROWS],
            ),
            "",
            "## 9. Dataset Curation",
            "",
            "The curated fused8 dataset separates repeat-stable targets from unstable and single-run-only candidates. That separation is deliberate: single-run-only rows can guide analysis and mining, but should not be used as direct SFT targets without review.",
            "",
            _markdown_table(
                ["Split", "Rows", "Intended use", "Train now?"],
                [[*row] for row in CURATED_DATASET_ROWS],
            ),
            "",
            "## 10. Failure Modes",
            "",
            "- Correct but slow kernels were the dominant model failure mode after prompt hardening.",
            "- Template guidance often preserved correctness but added wrapper or structural overhead in earlier runs.",
            "- Non-power-of-two Triton template variants caused compile failures until variant validation rejected them.",
            "- Qwen 7B zero-shot produced mostly invalid or slow candidates.",
            "- Qwen 14B has no model-quality result because serving failed with `No space left on device` during download/cache.",
            "",
            "## 11. Discussion",
            "",
            "The most important result is not that one model wins. It is that correctness and speed are separable. Once prompts and policy checks made correctness reliable, speed remained hard. Deterministic templates are not just baselines; they are a floor that model outputs must beat. Template context is useful for producing optimization data, but current models do not automatically preserve fast structure.",
            "",
            "## 12. Limitations",
            "",
            "- Internal fused8 benchmark only; not KernelBench.",
            "- Small task set and a single GPU class for the reported campaign.",
            "- No Nsight or hardware-counter profiling yet.",
            "- No fine-tuning and no RL.",
            "- Some numbers are provided-run summaries when the full RunPod artifact is not present in this workspace.",
            "- No SOTA claim.",
            "",
            "## 13. Next Work",
            "",
            "The next technical step is not training immediately. The project should first package the curated fused8 data, review stable-fast candidates manually, and compare a stronger local/open model such as Qwen 14B only after provisioning enough disk/cache. After that, move toward a small KernelBench L1 subset and prepare SFT data from repeat-stable targets and optimization pairs.",
            "",
            "## 14. Reproducibility Appendix",
            "",
            "See `reports/reproducibility.md` for exact command flows. The table sources are in `reports/tables/`. Run artifacts that are not present in the local checkout are listed explicitly in `reports/artifact_index.md`.",
            "",
        ]
    )


def _reproducibility_guide() -> str:
    return "\n".join(
        [
            "# OpenKernelForge Reproducibility Guide",
            "",
            "This guide reproduces the harness and internal fused8 workflow. Real Triton performance results require a CUDA GPU with Triton installed.",
            "",
            "## 1. Install",
            "",
            "```bash",
            "python -m pip install -e .",
            "pytest -q",
            "```",
            "",
            "## 2. Environment Check",
            "",
            "```bash",
            "python -m openkernelforge.cli env-check",
            "```",
            "",
            "For true Triton benchmark results, the viability should be `TRITON_EXECUTION_OK`.",
            "",
            "## 3. Fake Smoke Run",
            "",
            "```bash",
            "python -m openkernelforge.cli smoke",
            "```",
            "",
            "Fake and dummy runs are harness checks only. They are not model benchmarks.",
            "",
            "## 4. Fused8 Template Quick",
            "",
            "```bash",
            "python scripts/run_gpu_baseline_3tasks.py \\",
            "  --config configs/template_fused8_gpu_autotune_quick.yaml \\",
            "  --out-name template_fused8_gpu_quick",
            "```",
            "",
            "## 5. Fused8 Template Wide",
            "",
            "```bash",
            "python scripts/run_gpu_baseline_3tasks.py \\",
            "  --config configs/template_fused8_gpu_autotune_wide.yaml \\",
            "  --out-name template_fused8_gpu_wide",
            "```",
            "",
            "## 6. Repeatability",
            "",
            "```bash",
            "python -m openkernelforge.cli repeatability-report \\",
            "  --run-dir runs/<run> \\",
            "  --top-k 3 \\",
            "  --repeats 5",
            "```",
            "",
            "## 7. Optional Model Runs",
            "",
            "Gemini/OpenAI runs require API keys in environment variables only. Do not commit keys.",
            "",
            "```bash",
            "export GEMINI_API_KEY=<your-key>",
            "python scripts/run_gpu_baseline_3tasks.py --config configs/gemini_fused8_gpu_baseline.yaml --out-name gemini_fused8_gpu_baseline",
            "unset GEMINI_API_KEY",
            "```",
            "",
            "OpenAI cheap runs use `OPENAI_API_KEY` and should be kept small unless early results justify more spend.",
            "",
            "Local vLLM runs use the OpenAI-compatible local server path:",
            "",
            "```bash",
            "vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --host 0.0.0.0 --port 8000",
            "python scripts/run_local_model_fused8.py --config configs/qwen_fused8_gpu_baseline_cheap.yaml --out-name qwen_fused8_cheap",
            "```",
            "",
            "## 8. Curate Dataset",
            "",
            "```bash",
            "python -m openkernelforge.cli curate-fused8-dataset \\",
            "  --template-run runs/<template_run> \\",
            "  --gemini-run runs/<gemini_run> \\",
            "  --template-guided-run runs/<guided_run> \\",
            "  --out-dir datasets/fused8_curated_v1",
            "```",
            "",
            "## 9. Validate Curated Dataset",
            "",
            "```bash",
            "python -m openkernelforge.cli validate-curated-fused8 --dataset-dir datasets/fused8_curated_v1",
            "```",
            "",
            "## 10. Build Research Report",
            "",
            "```bash",
            "python scripts/build_phase14_report.py",
            "python scripts/check_research_artifacts.py",
            "```",
            "",
            "## Notes",
            "",
            "- GPU is required for real Triton correctness/performance claims.",
            "- API keys must be environment variables only.",
            "- `runs/` and `datasets/` should be reviewed before using them for training.",
            "- This is an internal fused8 workflow, not KernelBench and not a SOTA claim.",
            "",
        ]
    )


def _artifact_index(root: Path) -> str:
    lines = [
        "# OpenKernelForge Artifact Index",
        "",
        "This index lists important research artifacts and whether they are present in this workspace. Missing local artifacts are not inferred or fabricated.",
        "",
        "| Artifact | Canonical path | Local status | Description |",
        "| --- | --- | --- | --- |",
    ]
    for artifact in ARTIFACTS:
        local = root / artifact.local_path
        status = "present" if local.exists() else "not present in this workspace"
        lines.append(
            f"| {artifact.label} | `{artifact.canonical_path}` | {status} | {artifact.description} |"
        )
    generated = [
        ("technical report", "reports/openkernelforge_technical_report.md"),
        ("reproducibility guide", "reports/reproducibility.md"),
        ("artifact index", "reports/artifact_index.md"),
        ("CSV tables", "reports/tables/"),
    ]
    lines.extend(["", "## Generated Research Artifacts", ""])
    for label, path in generated:
        status = "present" if (root / path).exists() or path == "reports/artifact_index.md" else "not present in this workspace"
        lines.append(f"- {label}: `{path}` - {status}")
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _maybe_write_figures(figures: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    model_names = [row["baseline"] for row in MODEL_COMPARISON_ROWS]
    medians = [float(row["median_speedup_vs_eager"]) for row in MODEL_COMPARISON_ROWS]
    plt.figure(figsize=(10, 4))
    plt.bar(model_names, medians)
    plt.axhline(1.0, color="black", linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("median speedup vs eager")
    plt.tight_layout()
    plt.savefig(figures / "model_median_speedups.png", dpi=160)
    plt.close()

    task_names = [row[0] for row in FUSED8_TEMPLATE_ROWS]
    speedups = [float(row[1]) for row in FUSED8_TEMPLATE_ROWS]
    plt.figure(figsize=(10, 4))
    plt.bar(task_names, speedups)
    plt.axhline(1.0, color="black", linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("best single-run speedup vs eager")
    plt.tight_layout()
    plt.savefig(figures / "fused8_best_speedups.png", dpi=160)
    plt.close()
