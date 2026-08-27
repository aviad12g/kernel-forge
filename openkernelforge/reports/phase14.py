"""Research report and reproducibility artifact generation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TABLE_NAMES = [
    "benchmark_protocol.csv",
    "fused8_model_comparison.csv",
    "fused8_template_results.csv",
    "fused8_stable_winners.csv",
    "kernelbench_l1_pilot.csv",
    "uncertainty_extraction_status.csv",
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
    ("bias_relu", "1.029", "0.976", "no", "yes", "rigorous_cuda_event", "runs/20260520_155839"),
    ("sigmoid", "0.998", "0.934", "no", "yes", "rigorous_cuda_event", "runs/20260520_155839"),
    ("add_relu", "0.947", "0.938", "no", "yes", "rigorous_cuda_event", "runs/20260520_155839"),
    ("residual", "1.140", "1.023", "yes", "yes", "rigorous_cuda_event", "runs/20260520_155839"),
    ("bias_gelu", "1.473", "1.485", "yes", "yes", "rigorous_cuda_event", "runs/20260520_155839"),
    ("row_sum", "0.790", "0.674", "no", "yes", "rigorous_cuda_event", "runs/20260520_155839"),
    ("layernorm", "0.843", "0.791", "no", "yes", "rigorous_cuda_event", "runs/20260520_155839"),
    ("rmsnorm", "1.674", "1.452", "yes", "yes", "rigorous_cuda_event", "runs/20260520_155839"),
]

MODEL_COMPARISON_ROWS = [
    {
        "baseline": "template",
        "candidates": "160",
        "verified": "160/160",
        "median_speedup_vs_eager": "0.945",
        "median_speedup_vs_compile": "1.079",
        "uncertainty": "not preserved",
        "repeat_stable_wins": "residual, bias_gelu, rmsnorm",
        "conclusion": "strongest overall floor",
    },
    {
        "baseline": "Gemini",
        "candidates": "24",
        "verified": "23/24",
        "median_speedup_vs_eager": "0.923",
        "median_speedup_vs_compile": "1.863",
        "uncertainty": "not preserved",
        "repeat_stable_wins": "bias_gelu, rmsnorm",
        "conclusion": "strongest model correctness; stable wins below deterministic template medians",
    },
    {
        "baseline": "OpenAI mini",
        "candidates": "24",
        "verified": "12/24",
        "median_speedup_vs_eager": "0.888",
        "median_speedup_vs_compile": "1.835",
        "uncertainty": "not preserved",
        "repeat_stable_wins": "residual",
        "conclusion": "weaker correctness; one stable win over deterministic template repeat median",
    },
    {
        "baseline": "legacy model rows",
        "candidates": "various",
        "verified": "various",
        "median_speedup_vs_eager": "legacy",
        "median_speedup_vs_compile": "legacy",
        "uncertainty": "legacy",
        "repeat_stable_wins": "legacy timing",
        "conclusion": "historical context only; not primary paper-facing comparison",
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

BENCHMARK_PROTOCOL_ROWS = [
    {
        "study": "fused8 template/model study",
        "timing": "CUDA events",
        "repeats": "120",
        "sessions": "3",
        "cache_flush": "128 MB write",
        "compiler_baseline": "compile max-autotune",
        "candidate_source": "template, Gemini, OpenAI mini",
    },
    {
        "study": "Historical KernelBench L1 adapter output",
        "timing": "CUDA events",
        "repeats": "100/120",
        "sessions": "3",
        "cache_flush": "enabled",
        "compiler_baseline": "compile max-autotune",
        "candidate_source": "affected one-shot plus repair artifacts",
    },
]

KERNELBENCH_L1_PILOT_ROWS = [
    {
        "stage": "baseline validation",
        "tasks_or_candidates": "20 selected tasks",
        "verified": "20 eager / 20 compile timed",
        "benchmarked": "20",
        "stable_speedups": "not applicable",
        "uncertainty": "not applicable",
        "main_conclusion": "historical timing is provisional after adapter audit",
        "evidence_status": "historical_adapter_output_only",
    },
    {
        "stage": "one-shot Gemini",
        "tasks_or_candidates": "20 candidates",
        "verified": "3/20",
        "benchmarked": "3",
        "stable_speedups": "CE 1.992x [IQR 0.0022]; Triplet 4.176x [IQR 0.0005]",
        "uncertainty": "speedup CI not preserved; timing CI preserved in JSON",
        "main_conclusion": "affected evaluator output; not model accuracy",
        "evidence_status": "historical_adapter_output_only",
    },
    {
        "stage": "repair1",
        "tasks_or_candidates": "8 selected repairs",
        "verified": "1/8",
        "benchmarked": "1",
        "stable_speedups": "KLDiv 1.843x [IQR 0.0003]",
        "uncertainty": "speedup CI not preserved; timing CI preserved in JSON",
        "main_conclusion": "affected evaluator output; not repair evidence",
        "evidence_status": "historical_adapter_output_only",
    },
    {
        "stage": "combined after repair",
        "tasks_or_candidates": "20 tasks + 8 repairs",
        "verified": "4 unique tasks",
        "benchmarked": "4",
        "stable_speedups": "3 stable wins",
        "uncertainty": "see rows above",
        "main_conclusion": "no correctness or speed claim pending corrected rerun",
        "evidence_status": "historical_adapter_output_only",
    }
]

STABLE_WINNER_ROWS = [
    ("bias_relu", "none", "n/a", "0.976", "not preserved", "template single-run 1.029", "single-run template win fell below eager", "rigorous_cuda_event", "runs/20260520_155839"),
    ("sigmoid", "none", "n/a", "0.997", "std 0.029, CV 0.030", "Gemini 0.997", "nearest model remained below eager", "rigorous_cuda_event", "runs/20260520_163344"),
    ("add_relu", "none", "n/a", "0.968", "std 0.003, CV 0.003", "Gemini 0.968", "nearest model remained below eager", "rigorous_cuda_event", "runs/20260520_163344"),
    ("residual", "OpenAI mini", "llm", "1.074", "std 0.048, CV 0.045", "template 1.023", "OpenAI mini is the only model-over-template stable win", "rigorous_cuda_event", "runs/20260520_163607"),
    ("bias_gelu", "template", "template", "1.485", "not preserved", "Gemini 1.387", "template remains stronger than Gemini", "rigorous_cuda_event", "runs/20260520_155839"),
    ("row_sum", "none", "n/a", "0.674", "not preserved", "Gemini 0.646", "all verified candidates below eager", "rigorous_cuda_event", "runs/20260520_155839"),
    ("layernorm", "none", "n/a", "0.791", "not preserved", "Gemini 0.785", "all verified candidates below eager", "rigorous_cuda_event", "runs/20260520_155839"),
    ("rmsnorm", "template", "template", "1.452", "not preserved", "Gemini 1.415", "template remains strongest by repeat median", "rigorous_cuda_event", "runs/20260520_155839"),
]

UNCERTAINTY_STATUS_ROWS = [
    {
        "study": "fused8",
        "source": row["baseline"],
        "task": "aggregate",
        "metric": "median_speedup_vs_eager",
        "median_available": "yes",
        "iqr_available": "no",
        "ci_available": "no",
        "std_cv_available": "no",
        "artifact_source": "reports/tables/fused8_model_comparison.csv",
        "notes": "local package contains imported medians and labels; p25/p75 and bootstrap intervals are not preserved",
    }
    for row in MODEL_COMPARISON_ROWS[:3]
] + [
    {
        "study": "fused8",
        "source": source,
        "task": task,
        "metric": "repeat_median_speedup_vs_eager",
        "median_available": "yes",
        "iqr_available": "no",
        "ci_available": "no",
        "std_cv_available": "yes" if uncertainty.startswith("std") else "no",
        "artifact_source": run_dir if uncertainty.startswith("std") else "reports/tables/fused8_stable_winners.csv",
        "notes": uncertainty if uncertainty.startswith("std") else "repeat median preserved; interval samples not present in this local workspace",
    }
    for task, _winner, source, _repeat, uncertainty, _comparison, _interpretation, _timing_source, run_dir in STABLE_WINNER_ROWS
] + [
    {
        "study": "KernelBench L1 pilot",
        "source": "Gemini one-shot",
        "task": "CrossEntropyLoss",
        "metric": "speedup_vs_eager",
        "median_available": "yes",
        "iqr_available": "yes",
        "ci_available": "no",
        "std_cv_available": "no",
        "artifact_source": "runs/20260520_202314/results.jsonl",
        "notes": "historical affected-adapter field; not performance evidence",
    },
    {
        "study": "KernelBench L1 pilot",
        "source": "Gemini one-shot",
        "task": "TripletMarginLoss",
        "metric": "speedup_vs_eager",
        "median_available": "yes",
        "iqr_available": "yes",
        "ci_available": "no",
        "std_cv_available": "no",
        "artifact_source": "runs/20260520_202314/results.jsonl",
        "notes": "historical affected-adapter field; not performance evidence",
    },
    {
        "study": "KernelBench L1 pilot",
        "source": "Gemini one-shot",
        "task": "Matmul_with_diagonal_matrices",
        "metric": "speedup_vs_eager",
        "median_available": "yes",
        "iqr_available": "yes",
        "ci_available": "no",
        "std_cv_available": "no",
        "artifact_source": "runs/20260520_202314/results.jsonl",
        "notes": "historical affected-adapter field; not correctness or performance evidence",
    },
    {
        "study": "KernelBench L1 repair1",
        "source": "Gemini repair",
        "task": "KLDivLoss",
        "metric": "speedup_vs_eager",
        "median_available": "yes",
        "iqr_available": "yes",
        "ci_available": "no",
        "std_cv_available": "no",
        "artifact_source": "runs/20260520_213128/results.jsonl",
        "notes": "historical affected-adapter field; not repair or performance evidence",
    },
    {
        "study": "KernelBench L1 pilot",
        "source": "Gemini one-shot",
        "task": "verified candidates",
        "metric": "candidate/eager/compile_ms_summary",
        "median_available": "yes",
        "iqr_available": "yes",
        "ci_available": "yes",
        "std_cv_available": "yes",
        "artifact_source": "runs/20260520_202314/results.jsonl",
        "notes": "historical timing summaries preserved, but invalid reference lifecycle prevents interpretation",
    },
    {
        "study": "KernelBench L1 repair1",
        "source": "Gemini repair",
        "task": "KLDivLoss",
        "metric": "candidate/eager/compile_ms_summary",
        "median_available": "yes",
        "iqr_available": "yes",
        "ci_available": "yes",
        "std_cv_available": "yes",
        "artifact_source": "runs/20260520_213128/results.jsonl",
        "notes": "historical timing summary preserved, but invalid reference lifecycle prevents interpretation",
    },
]

ARTIFACTS = [
    Artifact(
        "rigorous deterministic fused8 template",
        "/workspace/openkernelforge/runs/20260520_155839",
        "runs/20260520_155839",
        "160-candidate CUDA-event deterministic template run",
    ),
    Artifact(
        "rigorous Gemini fused8 baseline",
        "/workspace/openkernelforge/runs/20260520_163344",
        "runs/20260520_163344",
        "24-candidate rigorous Gemini baseline",
    ),
    Artifact(
        "rigorous OpenAI mini fused8 baseline",
        "/workspace/openkernelforge/runs/20260520_163607",
        "runs/20260520_163607",
        "24-candidate rigorous OpenAI mini baseline",
    ),
    Artifact(
        "rigorous fused8 model comparison",
        "/workspace/openkernelforge/runs/rigorous_fused8_model_comparison.md",
        "runs/rigorous_fused8_model_comparison.md",
        "rigorous template/Gemini/OpenAI mini comparison report",
    ),
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
    Artifact(
        "KernelBench safe baseline validation",
        "/workspace/openkernelforge/runs/20260520_181052",
        "runs/20260520_181052",
        "20-task feasible-subset eager and torch.compile baseline validation",
    ),
    Artifact(
        "KernelBench Gemini candidate pilot",
        "/workspace/openkernelforge/runs/20260520_202314",
        "runs/20260520_202314",
        "20-task capped Gemini candidate pilot with failure taxonomy",
    ),
    Artifact(
        "KernelBench candidate failure analysis",
        "/workspace/openkernelforge/runs/20260520_202314/kernelbench_candidate_failure_analysis.md",
        "runs/20260520_202314/kernelbench_candidate_failure_analysis.md",
        "failure taxonomy report for 17 failed KernelBench candidates",
    ),
    Artifact(
        "KernelBench failure taxonomy JSON",
        "/workspace/openkernelforge/runs/20260520_202314/kernelbench_failure_taxonomy.json",
        "runs/20260520_202314/kernelbench_failure_taxonomy.json",
        "machine-readable KernelBench candidate failure taxonomy",
    ),
    Artifact(
        "KernelBench repair subset",
        "/workspace/openkernelforge/runs/20260520_202314/kernelbench_repair_subset.md",
        "runs/20260520_202314/kernelbench_repair_subset.md",
        "selected 8 high-repairability tasks for one capped repair pass",
    ),
    Artifact(
        "KernelBench Gemini repair pass",
        "/workspace/openkernelforge/runs/20260520_213128",
        "runs/20260520_213128",
        "8-task capped Gemini repair iteration",
    ),
    Artifact(
        "KernelBench repair comparison",
        "/workspace/openkernelforge/runs/kernelbench_gemini_repair1_comparison.md",
        "runs/kernelbench_gemini_repair1_comparison.md",
        "original-vs-repair KernelBench comparison report",
    ),
    Artifact(
        "KernelBench memory-safe selection config",
        "/workspace/openkernelforge/configs/kernelbench_l1_20task_rigorous_safe.yaml",
        "configs/kernelbench_l1_20task_rigorous_safe.yaml",
        "memory-capped 20-task baseline validation config",
    ),
    Artifact(
        "KernelBench Gemini pilot config",
        "/workspace/openkernelforge/configs/kernelbench_l1_20task_gemini_rigorous.yaml",
        "configs/kernelbench_l1_20task_gemini_rigorous.yaml",
        "20-task capped Gemini candidate pilot config",
    ),
    Artifact(
        "KernelBench Gemini repair config",
        "/workspace/openkernelforge/configs/kernelbench_l1_20task_gemini_repair1.yaml",
        "configs/kernelbench_l1_20task_gemini_repair1.yaml",
        "8-task capped Gemini repair config",
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
            tables / "benchmark_protocol.csv",
            [
                "study",
                "timing",
                "repeats",
                "sessions",
                "cache_flush",
                "compiler_baseline",
                "candidate_source",
            ],
            BENCHMARK_PROTOCOL_ROWS,
        ),
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
                "uncertainty",
                "above_eager_stable",
                "above_torch_compile",
                "timing_source",
                "run_dir",
            ],
            [
                {
                    "task": task,
                    "best_single_run_speedup_vs_eager": speedup,
                    "repeat_median_speedup_vs_eager": repeat,
                    "uncertainty": "not preserved",
                    "above_eager_stable": stable,
                    "above_torch_compile": compile_win,
                    "timing_source": timing_source,
                    "run_dir": run_dir,
                }
                for task, speedup, repeat, stable, compile_win, timing_source, run_dir in FUSED8_TEMPLATE_ROWS
            ],
        ),
        _write_csv(
            tables / "fused8_model_comparison.csv",
            [
                "baseline",
                "candidates",
                "verified",
                "median_speedup_vs_eager",
                "median_speedup_vs_compile",
                "uncertainty",
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
            tables / "kernelbench_l1_pilot.csv",
            [
                "stage",
                "tasks_or_candidates",
                "verified",
                "benchmarked",
                "stable_speedups",
                "uncertainty",
                "main_conclusion",
                "evidence_status",
            ],
            KERNELBENCH_L1_PILOT_ROWS,
        ),
        _write_csv(
            tables / "fused8_stable_winners.csv",
            [
                "task",
                "stable_winner",
                "source_type",
                "repeat_median_speedup_vs_eager",
                "uncertainty",
                "closest_model_template_comparison",
                "interpretation",
                "timing_source",
                "run_dir",
            ],
            [
                {
                    "task": task,
                    "stable_winner": winner,
                    "source_type": source,
                    "repeat_median_speedup_vs_eager": repeat,
                    "uncertainty": uncertainty,
                    "closest_model_template_comparison": comparison,
                    "interpretation": interpretation,
                    "timing_source": timing_source,
                    "run_dir": run_dir,
                }
                for task, winner, source, repeat, uncertainty, comparison, interpretation, timing_source, run_dir in STABLE_WINNER_ROWS
            ],
        ),
        _write_csv(
            tables / "uncertainty_extraction_status.csv",
            [
                "study",
                "source",
                "task",
                "metric",
                "median_available",
                "iqr_available",
                "ci_available",
                "std_cv_available",
                "artifact_source",
                "notes",
            ],
            UNCERTAINTY_STATUS_ROWS,
        ),
    ]
    return written


def _technical_report() -> str:
    return "\n".join(
        [
            "# OpenKernelForge: Repeatability-Aware Evaluation for LLM-Generated Triton Kernels",
            "",
            "## 1. Abstract",
            "",
            "Generated GPU kernels need correctness and repeat-stable speed, not merely plausible source code. OpenKernelForge evaluates Triton candidates with policy checks, correctness tests, CUDA-event timing, cache-state perturbation, repeated sessions, session-level summaries, and compiler baselines. In fused8, Gemini and OpenAI mini reach roughly 1.8x median speedup over `torch.compile max-autotune` but remain below PyTorch eager at median, while deterministic templates remain the strongest floor. A post-hoc audit of the historical KernelBench pilot found an invalid stateful-task contract and reference construction inside timed calls. Those external rows are retained as evaluator-audit artifacts but excluded from correctness and performance claims pending corrected CUDA revalidation.",
            "",
            "## 2. Motivation",
            "",
            "Kernel generation agents need more than pass/fail execution. Useful systems need reproducible prompts, raw responses, candidate sources, policy checks, correctness traces, timing data, repeatability, and structured failure labels. OpenKernelForge was built to make those artifacts first-class so generated-kernel claims are grounded in repeatable evidence rather than isolated fast samples.",
            "",
            "## 3. System Overview",
            "",
            "- Task layer: PyTorch references, deterministic inputs, shape metadata, tolerances, and prompt hints.",
            "- Agent layer: dummy, fake, OpenAI-compatible, local vLLM-compatible, and deterministic template agents.",
            "- Harness layer: candidate extraction, conservative AST policy checks, trusted in-process loading, verifier, benchmarker, and JSONL logging.",
            "- Reporting layer: summaries, run analysis, failure taxonomy, repeatability reports, fused8 reports, and dataset curation.",
            "- Artifact layer: prompt files, raw responses, candidate source, logs, environment probes, datasets, and human-readable reports.",
            "",
            "## 4. Benchmark Tasks",
            "",
            "The project started with a three-task sandbox: `vector_add`, `relu`, and `bias_relu`. That sandbox validated the harness but showed that isolated elementwise tasks are poor standalone performance targets. The project then moved to an internal fused8 benchmark: `bias_relu`, `sigmoid`, `add_relu`, `residual`, `bias_gelu`, `row_sum`, `layernorm`, and `rmsnorm`. These fused workloads are better aligned with Triton launch amortization and realistic kernel-generation behavior.",
            "",
            "## 5. Methods",
            "",
            "Fused8 candidates expose `forward(*args)`. Official KernelBench tasks materialize one seeded `get_init_inputs()` snapshot and construct persistent reference `Model` and candidate `ModelNew` instances from it before verification and timing; every official `Model` task rejects free functions. Candidates pass conservative AST guardrails before trusted in-process loading. The current rigorous path records CUDA-event timing, 30 warmup iterations, 120 measured samples per session, three same-process sessions with rotating measurement order, session-level summaries, cache-state perturbation, materialized compile accounting, and runtime-only comparison against PyTorch eager and `torch.compile max-autotune`. AST checks are not an operating-system sandbox.",
            "",
            "## 6. Model and Template Baselines",
            "",
            "- Deterministic templates sweep block size, warps, stages, allocation policy, contiguity policy, and shape specialization.",
            "- Gemini rigorous fused8 used 24 generated candidates and verified 23/24.",
            "- OpenAI mini rigorous fused8 used 24 generated candidates and verified 12/24.",
            "- Earlier Gemini template-guided, OpenAI, GPT-5.5, and Qwen runs are legacy timing context, not primary paper-facing results.",
            "- Qwen 7B local zero-shot was weak under the cheap fused8 protocol; Qwen 14B was not evaluated because serving failed due disk/cache capacity.",
            "",
            "## 7. Results",
            "",
            "Provenance note: the paper-facing rigorous fused8 numbers are from RunPod artifacts. `reports/artifact_index.md` records which canonical artifacts are present locally.",
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
            "These are the current paper-facing deterministic-template numbers from the rigorous CUDA-event run `runs/20260520_155839`. The older 2076-candidate template-wide table is legacy timing.",
            "",
            _markdown_table(
                ["Task", "Best single-run", "Repeat median", "Stable above eager", "Above torch.compile"],
                [[task, speedup, repeat, stable, compile_win] for task, speedup, repeat, stable, compile_win, _, _ in FUSED8_TEMPLATE_ROWS],
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
                    "Median compile speedup",
                    "Uncertainty",
                    "Repeat-stable wins",
                    "Conclusion",
                ],
                [
                    [
                        row["baseline"],
                        row["candidates"],
                        row["verified"],
                        row["median_speedup_vs_eager"],
                        row["median_speedup_vs_compile"],
                        row["uncertainty"],
                        row["repeat_stable_wins"],
                        row["conclusion"],
                    ]
                    for row in MODEL_COMPARISON_ROWS
                ],
            ),
            "",
            "## 8. Repeatability Analysis",
            "",
            "Repeatability changed several conclusions. In the three-task sandbox, single-run wins for `bias_relu` did not survive repeat benchmarking. In rigorous fused8, deterministic `bias_relu` was also a single-run-only win: it reached 1.029x in the original run and fell to 0.976x repeat median. The top-1 LLM above-eager wins in the rigorous Gemini and OpenAI mini runs did survive repeatability, so the strongest broad LLM-fragility headline is not supported by this sample. Single-run wins remain useful search signals, but they are not sufficient evidence for benchmark claims.",
            "",
            "### Stable Winners By Task",
            "",
            _markdown_table(
                ["Task", "Stable winner", "Source type", "Repeat median", "Uncertainty", "Closest comparison", "Interpretation"],
                [
                    [task, winner, source, repeat, uncertainty, comparison, interpretation]
                    for task, winner, source, repeat, uncertainty, comparison, interpretation, _, _ in STABLE_WINNER_ROWS
                ],
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
            "The most important result is not that one model wins. It is that correctness, single-run speed, repeat-stable speed, and compiler-baseline performance are separable. Gemini is strong on fused8 correctness, OpenAI mini finds the stable `residual` winner despite weaker correctness, and templates remain the strongest overall floor. Generated kernels may beat weak compiler-generated paths while still losing to library-specialized eager kernels. The evaluation layer is therefore the main contribution: generated kernels should be discussed with repeatability labels and compiler baselines, not isolated timing samples.",
            "",
            "## 12. Limitations",
            "",
            "- Historical KernelBench pilot and repair rows are provisional because the old adapter violated state and timing contracts; corrected CUDA revalidation is outstanding.",
            "- Small task set and a single GPU class for the reported campaign.",
            "- Small rigorous model budgets: 24 generated candidates for Gemini and 24 for OpenAI mini.",
            "- No Nsight or hardware-counter profiling yet.",
            "- No fine-tuning and no RL.",
            "- API model behavior can change over time.",
            "- This is not a SOTA claim.",
            "",
            "## 13. Historical KernelBench Adapter Audit",
            "",
            "The preserved KernelBench artifacts come from the official checkout at commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`. The historical evaluator recorded 3/20 one-shot verifications and 1/8 repair verifications, but a post-hoc audit found that it replaced the official `ModelNew` lifecycle with free-function candidates and reconstructed reference modules inside timed calls. Parameterized tasks additionally lost access to initialized state. The current AST policy also rejects 9/20 one-shot sources that the historical policy accepted. These counts are retained for auditability and failure analysis, not as estimates of Gemini correctness or KernelBench performance. The corrected adapter requires a new CUDA run.",
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
            "python scripts/build_paper_pdf.py",
            "python scripts/check_research_artifacts.py",
            "```",
            "",
            "## 11. Corrected KernelBench Validation Workflow",
            "",
            "The official KernelBench repository was recorded at commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`. The corrected adapter uses official `ModelNew` state semantics, persistent equally seeded reference and candidate modules, and a conservative lower-bound memory filter. The historical runs listed below predate those corrections and must not be treated as reproduced evidence.",
            "",
            "Clone KernelBench:",
            "",
            "```bash",
            "git clone https://github.com/ScalingIntelligence/KernelBench.git /workspace/KernelBench",
            "cd /workspace/KernelBench",
            "git checkout 423217d9fda91e0c2d67e4a43bf62f96f6d104f1",
            "```",
            "",
            "Run safe baseline validation:",
            "",
            "```bash",
            "python -m openkernelforge.cli kernelbench-l1-check \\",
            "  --config configs/kernelbench_l1_20task_rigorous_safe.yaml \\",
            "  --kernelbench-dir /workspace/KernelBench",
            "```",
            "",
            "Run the capped Gemini candidate pilot. API keys must be environment variables only; never commit them.",
            "",
            "```bash",
            "export GEMINI_API_KEY=<key>",
            "python -m openkernelforge.cli kernelbench-l1-check \\",
            "  --config configs/kernelbench_l1_20task_gemini_rigorous.yaml \\",
            "  --kernelbench-dir /workspace/KernelBench",
            "unset GEMINI_API_KEY",
            "```",
            "",
            "Analyze failed generated candidates and select the one-pass repair subset:",
            "",
            "```bash",
            "python scripts/analyze_kernelbench_candidate_failures.py \\",
            "  --run-dir runs/20260520_202314 \\",
            "  --max-repair 8",
            "```",
            "",
            "Run the capped repair pass:",
            "",
            "```bash",
            "export GEMINI_API_KEY=<key>",
            "python -m openkernelforge.cli kernelbench-l1-check \\",
            "  --config configs/kernelbench_l1_20task_gemini_repair1.yaml \\",
            "  --kernelbench-dir /workspace/KernelBench",
            "unset GEMINI_API_KEY",
            "```",
            "",
            "The original-vs-repair comparison is recorded at:",
            "",
            "```text",
            "runs/kernelbench_gemini_repair1_comparison.md",
            "```",
            "",
            "Historical status: the affected evaluator recorded 3/20 one-shot verifications and 1/8 repair verifications. Those rows are preserved under their original run IDs, but no KernelBench correctness or speedup claim is supported until the corrected workflow is rerun on CUDA. Run `python scripts/audit_historical_kernelbench_candidates.py` to reproduce the static policy re-audit without executing candidates.",
            "",
            "## Notes",
            "",
            "- GPU is required for real Triton correctness/performance claims.",
            "- API keys must be environment variables only.",
            "- `runs/` and `datasets/` should be reviewed before using them for training.",
            "- Internal fused8 is the supported controlled study; historical KernelBench rows are evaluator-audit artifacts pending corrected CUDA revalidation.",
            "",
        ]
    )


def _artifact_index(root: Path) -> str:
    lines = [
        "# OpenKernelForge Artifact Index",
        "",
        "This index lists important research artifacts and whether they are present in this workspace. Missing local artifacts are not inferred or fabricated.",
        "",
        "- KernelBench repo path used on RunPod: `/workspace/KernelBench`",
        "- KernelBench commit: `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`",
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
        ("paper PDF", "paper/openkernelforge_paper.pdf"),
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
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
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
        from matplotlib.patches import FancyBboxPatch
    except Exception:
        return

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    palette = {
        "template": "#4c78a8",
        "Gemini": "#54a24b",
        "OpenAI mini": "#f58518",
        "neutral": "#6b7280",
        "danger": "#d65f5f",
        "gold": "#b79a20",
    }

    # Figure 1: clean pipeline diagram.
    labels = [
        "Prompt or\nTemplate",
        "Candidate\nSource",
        "Policy\nCheck",
        "Correctness\nVerifier",
        "CUDA-Event\nBenchmark",
        "Repeatability\nLabel",
        "Report and\nDataset",
    ]
    fig, ax = plt.subplots(figsize=(13.4, 2.5))
    ax.set_axis_off()
    positions = [(0.02 + i * 0.14, 0.43) for i in range(len(labels))]
    for idx, ((x, y), label) in enumerate(zip(positions, labels)):
        box = FancyBboxPatch(
            (x, y),
            0.115,
            0.34,
            boxstyle="round,pad=0.02,rounding_size=0.015",
            linewidth=1.1,
            edgecolor="#1f2937",
            facecolor="#f8fafc",
        )
        ax.add_patch(box)
        ax.text(x + 0.0575, y + 0.17, label, ha="center", va="center", wrap=True, fontsize=10.8, color="#111827")
        if idx < len(labels) - 1:
            nx, ny = positions[idx + 1]
            start = (x + 0.115, y + 0.17)
            end = (nx, ny + 0.17)
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(arrowstyle="->", color="#1f2937", lw=1.25),
            )
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.22, 0.9)
    ax.set_title("OpenKernelForge evaluation pipeline", pad=5, fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(figures / "openkernelforge_pipeline.png", dpi=180)
    plt.close(fig)

    # Figure 2: repeat-stable winners by task/source.
    stable = [
        ("residual", "OpenAI mini", 1.074),
        ("bias_gelu", "template", 1.485),
        ("rmsnorm", "template", 1.452),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.bar([row[0] for row in stable], [row[2] for row in stable], color=[palette[row[1]] for row in stable])
    ax.axhline(1.0, color="#111827", lw=1.0)
    ax.set_ylabel("repeat median speedup vs eager")
    ax.set_title("Repeat-stable fused8 winners")
    ax.set_ylim(0, max(row[2] for row in stable) * 1.22)
    for i, (_, source, value) in enumerate(stable):
        ax.text(i, value + 0.035, f"{value:.3f}x\n{source}", ha="center", va="bottom", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(figures / "fused8_stable_speedups.png", dpi=180)
    fig.savefig(figures / "fused8_repeat_stable_speedups.png", dpi=180)
    plt.close(fig)

    # Figure 2b: single-run vs repeatability callout for bias_relu.
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    values = [1.029, 0.976]
    labels = ["single-run", "repeat median"]
    bars = ax.bar(labels, values, color=["#9ca3af", palette["danger"]], width=0.55)
    ax.axhline(1.0, color="#111827", lw=1.0)
    ax.set_ylim(0.90, 1.06)
    ax.set_ylabel("speedup vs eager")
    ax.set_title("bias_relu: single-run win becomes single-only")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}x", ha="center", va="bottom", fontsize=10.5)
    ax.text(1, 0.918, "SINGLE_RUN_ONLY_WIN", ha="center", va="center", fontsize=9.5, color="#991b1b", weight="bold")
    fig.tight_layout()
    fig.savefig(figures / "bias_relu_single_run_flip.png", dpi=180)
    plt.close(fig)

    # Figure 3: source summary.
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.set_axis_off()
    ax.set_title("Fused8 source summary", fontsize=13, weight="bold", pad=8)
    table_rows = [
        ["Source", "Candidates", "Verified", "Stable wins"],
        ["template", "160", "160/160", "3"],
        ["Gemini", "24", "23/24", "2"],
        ["OpenAI mini", "24", "12/24", "1"],
    ]
    table = ax.table(cellText=table_rows, cellLoc="center", loc="center", colWidths=[0.30, 0.22, 0.24, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(weight="bold", color="#111827")
        elif col == 0:
            cell.set_facecolor("#f8fafc")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(figures / "fused8_source_summary.png", dpi=180)
    plt.close(fig)

    # Figure 4: KernelBench pilot funnel.
    funnel_labels = ["selected\nTasks", "candidates\nGenerated", "one-shot\nVerified", "one-shot\nStable", "repairs\nAttempted", "repair\nVerified", "total\nStable"]
    funnel_values = [20, 20, 3, 2, 8, 1, 3]
    funnel_colors = [palette["neutral"], palette["neutral"], palette["Gemini"], palette["Gemini"], palette["gold"], palette["gold"], palette["Gemini"]]
    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    bars = ax.bar(funnel_labels, funnel_values, color=funnel_colors)
    ax.set_title("Historical KernelBench adapter output funnel")
    ax.set_ylabel("count")
    ax.set_ylim(0, 22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    for bar, value in zip(bars, funnel_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, str(value), ha="center", va="bottom", fontsize=10.5)
    plt.tight_layout()
    fig.savefig(figures / "kernelbench_pilot_funnel.png", dpi=180)
    plt.close()

    # Figure 5: KernelBench failure taxonomy.
    failure_labels = ["numeric", "timeout/OOM", "runtime", "Triton compile"]
    failure_values = [9, 4, 2, 2]
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    bars = ax.bar(failure_labels, failure_values, color=[palette["danger"], palette["gold"], palette["neutral"], "#8b5cf6"])
    ax.set_title("Historical KernelBench verifier taxonomy")
    ax.set_ylabel("failed candidates")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    for bar, value in zip(bars, failure_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18, str(value), ha="center", va="bottom", fontsize=10.5)
    plt.tight_layout()
    fig.savefig(figures / "kernelbench_failure_taxonomy.png", dpi=180)
    plt.close()

    # Backward-compatible report figures retained for existing artifact checks.
    model_rows = []
    for model_row in MODEL_COMPARISON_ROWS:
        try:
            model_rows.append(
                (model_row["baseline"], float(model_row["median_speedup_vs_eager"]))
            )
        except (TypeError, ValueError):
            continue
    if model_rows:
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        ax.bar([row[0] for row in model_rows], [row[1] for row in model_rows], color="#6b7280")
        ax.axhline(1.0, color="#111827", linewidth=1)
        ax.tick_params(axis="x", labelrotation=20)
        ax.set_ylabel("median speedup vs eager")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures / "model_median_speedups.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar([row[0] for row in FUSED8_TEMPLATE_ROWS], [float(row[1]) for row in FUSED8_TEMPLATE_ROWS], color="#6b7280")
    ax.axhline(1.0, color="#111827", linewidth=1)
    ax.tick_params(axis="x", labelrotation=25)
    ax.set_ylabel("best single-run speedup vs eager")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "fused8_best_speedups.png", dpi=160)
    plt.close(fig)
