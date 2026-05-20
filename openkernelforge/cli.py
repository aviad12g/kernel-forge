"""Command-line interface for OpenKernelForge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from openkernelforge.agents.backends import create_backend
from openkernelforge.config import RunConfig, load_config
from openkernelforge.datasets.export import export_dataset, validate_dataset
from openkernelforge.harness.runner import run_from_config
from openkernelforge.reports.analyze import analyze_run
from openkernelforge.reports.compare import compare_runs_markdown
from openkernelforge.reports.gpu_debrief import debrief_gpu_run
from openkernelforge.reports.fused8 import write_fused8_report
from openkernelforge.reports.fused8_curation import (
    curate_fused8_dataset,
    inspect_curated_fused8_dataset,
    validate_curated_fused8_dataset,
)
from openkernelforge.reports.focused_sweep import (
    write_focused_sweep_report,
    write_focused_sweep_seed_analysis,
)
from openkernelforge.reports.profiler_lite import write_profiler_lite_report
from openkernelforge.reports.final_3task import write_final_3task_report
from openkernelforge.reports.repeatability import write_repeatability_report
from openkernelforge.reports.review import review_real_run
from openkernelforge.reports.summarize import write_summary
from openkernelforge.reports.template_copy import write_template_copy_report
from openkernelforge.reports.template_report import write_template_autotune_report
from openkernelforge.utils.env_probe import format_environment_summary, probe_environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openkernelforge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("smoke", help="Run a small dummy-agent smoke test")

    run_parser = subparsers.add_parser("run", help="Run tasks from a YAML config")
    run_parser.add_argument("--config", required=True, help="Path to YAML config")
    run_parser.add_argument("--agent", choices=["dummy", "llm", "template"], help="Override agent type")
    run_parser.add_argument("--backend", help="Override LLM backend name")
    run_parser.add_argument("--model", help="Override model name")
    run_parser.add_argument("--base-url", help="Override OpenAI-compatible base URL")

    summarize_parser = subparsers.add_parser("summarize", help="Regenerate summary.md for a run")
    summarize_parser.add_argument("--run-dir", required=True, help="Path to runs/<timestamp>")

    inspect_parser = subparsers.add_parser("inspect-run", help="Print an existing run summary")
    inspect_parser.add_argument("--run-dir", required=True, help="Path to runs/<timestamp>")

    show_config_parser = subparsers.add_parser("show-config", help="Print a redacted resolved config")
    show_config_parser.add_argument("--config", required=True, help="Path to YAML config")

    check_backend_parser = subparsers.add_parser(
        "check-backend", help="Check configured model backend health"
    )
    check_backend_parser.add_argument("--config", required=True, help="Path to YAML config")

    compare_parser = subparsers.add_parser("compare-runs", help="Compare run directories")
    compare_parser.add_argument("run_dirs", nargs="+", help="Run directories to compare")

    analyze_parser = subparsers.add_parser("analyze-run", help="Analyze a run and write analysis.md")
    analyze_parser.add_argument("--run-dir", required=True, help="Path to runs/<timestamp>")

    export_parser = subparsers.add_parser("export-dataset", help="Export run data to JSONL datasets")
    export_parser.add_argument("--run-dir", required=True, help="Path to runs/<timestamp>")
    export_parser.add_argument("--out-dir", required=True, help="Output dataset directory")

    validate_parser = subparsers.add_parser("validate-dataset", help="Validate an exported dataset")
    validate_parser.add_argument("--dataset-dir", required=True, help="Dataset directory")

    review_parser = subparsers.add_parser("review-real-run", help="Write real_run_review.md for a run")
    review_parser.add_argument("--run-dir", required=True, help="Path to runs/<timestamp>")

    env_parser = subparsers.add_parser("env-check", help="Check local CUDA/Triton execution environment")
    env_parser.add_argument("--out", help="Optional path to write environment_probe.json")

    debrief_parser = subparsers.add_parser(
        "debrief-gpu-run", help="Write gpu_candidate_debrief.md for a GPU run"
    )
    debrief_parser.add_argument("--run-dir", required=True, help="Path to runs/<timestamp>")

    template_report_parser = subparsers.add_parser(
        "template-report", help="Write template_autotune_report.md for a template run"
    )
    template_report_parser.add_argument("--run-dir", required=True, help="Path to runs/<timestamp>")
    template_report_parser.add_argument(
        "--compare-run-dir",
        help="Optional LLM run directory to include in the report comparison",
    )

    profiler_parser = subparsers.add_parser(
        "profiler-lite", help="Write profiler_lite_report.md for a run"
    )
    profiler_parser.add_argument("--run-dir", required=True, help="Path to runs/<timestamp>")

    template_copy_report_parser = subparsers.add_parser(
        "template-copy-report", help="Write template_copy_report.md for a template-copy run"
    )
    template_copy_report_parser.add_argument("--run-dir", required=True, help="Path to runs/<timestamp>")

    focused_report_parser = subparsers.add_parser(
        "focused-sweep-report", help="Write focused_sweep_report.md for a focused template run"
    )
    focused_report_parser.add_argument("--run-dir", required=True, help="Path to runs/<timestamp>")
    focused_report_parser.add_argument("--shapeaware-run-dir")
    focused_report_parser.add_argument("--template-copy-wide-run-dir")

    focused_seed_parser = subparsers.add_parser(
        "focused-seed-analysis", help="Write focused sweep seed analysis"
    )
    focused_seed_parser.add_argument("--shapeaware-run-dir", required=True)
    focused_seed_parser.add_argument("--template-copy-wide-run-dir", required=True)
    focused_seed_parser.add_argument("--out", default="runs/focused_sweep_seed_analysis.md")

    repeatability_parser = subparsers.add_parser(
        "repeatability-report", help="Rebenchmark top candidates and write repeatability artifacts"
    )
    repeatability_parser.add_argument("--run-dir", required=True, help="Path to runs/<timestamp>")
    repeatability_parser.add_argument("--top-k", type=int, default=5)
    repeatability_parser.add_argument("--repeats", type=int, default=5)

    final_parser = subparsers.add_parser(
        "final-3task-report", help="Write final_3task_conclusion.md"
    )
    final_parser.add_argument("--base-template", required=True)
    final_parser.add_argument("--shapeaware", required=True)
    final_parser.add_argument("--template-copy-wide", required=True)
    final_parser.add_argument("--focused", required=True)
    final_parser.add_argument("--clean-focused", required=True)
    final_parser.add_argument("--out", default="runs/final_3task_conclusion.md")

    fused8_parser = subparsers.add_parser("fused8-report", help="Write fused8_report.md for a fused8 run")
    fused8_parser.add_argument("--run-dir", required=True, help="Path to runs/<timestamp>")

    curate_fused8_parser = subparsers.add_parser(
        "curate-fused8-dataset", help="Curate fused8 template/Gemini runs into repeatability-aware splits"
    )
    curate_fused8_parser.add_argument("--template-run", required=True)
    curate_fused8_parser.add_argument("--gemini-run", required=True)
    curate_fused8_parser.add_argument("--template-guided-run", required=True)
    curate_fused8_parser.add_argument("--out-dir", required=True)

    inspect_curated_parser = subparsers.add_parser(
        "inspect-curated-fused8", help="Write a human-readable inspection report for a curated fused8 dataset"
    )
    inspect_curated_parser.add_argument("--dataset-dir", required=True)

    validate_curated_parser = subparsers.add_parser(
        "validate-curated-fused8", help="Validate curated fused8 split integrity"
    )
    validate_curated_parser.add_argument("--dataset-dir", required=True)

    args = parser.parse_args(argv)

    if args.command == "smoke":
        config_path = Path("configs") / "smoke.yaml"
        config = load_config(config_path) if config_path.exists() else RunConfig()
        run_dir = run_from_config(config)
        print(f"Smoke run complete: {run_dir}")
        return 0

    if args.command == "run":
        config = load_config(args.config)
        if args.agent:
            config.agent.type = args.agent
        if args.backend:
            config.agent.backend = args.backend
        if args.model:
            config.agent.model = args.model
        if args.base_url:
            config.agent.base_url = args.base_url
        run_dir = run_from_config(config)
        print(f"Run complete: {run_dir}")
        return 0

    if args.command == "summarize":
        summary_path = write_summary(args.run_dir)
        print(f"Summary written: {summary_path}")
        return 0

    if args.command == "inspect-run":
        summary_path = Path(args.run_dir) / "summary.md"
        if not summary_path.exists():
            summary_path = write_summary(args.run_dir)
        print(summary_path.read_text(encoding="utf-8"))
        return 0

    if args.command == "show-config":
        config = load_config(args.config)
        print(yaml.safe_dump(config.to_safe_dict(), sort_keys=False))
        return 0

    if args.command == "check-backend":
        config = load_config(args.config)
        try:
            backend = create_backend(config.agent)
            generate_kwargs = {}
            if config.agent.temperature is not None:
                generate_kwargs["temperature"] = config.agent.temperature
            if config.agent.top_p is not None:
                generate_kwargs["top_p"] = config.agent.top_p
            response = backend.generate(
                "Return the word OK.",
                system="You are a backend health check. Return only OK.",
                **generate_kwargs,
            )
            if "OK" not in response.upper():
                print("Backend check failed: response did not contain OK")
                return 1
        except Exception as exc:
            print(f"Backend check failed: {exc}")
            return 1
        print("Backend check succeeded.")
        return 0

    if args.command == "compare-runs":
        print(compare_runs_markdown(args.run_dirs), end="")
        return 0

    if args.command == "analyze-run":
        analysis_path = analyze_run(args.run_dir)
        print(f"Analysis written: {analysis_path}")
        preview_lines = analysis_path.read_text(encoding="utf-8").splitlines()[:18]
        print("\n".join(preview_lines))
        return 0

    if args.command == "export-dataset":
        out_path = export_dataset(args.run_dir, args.out_dir)
        print(f"Dataset exported: {out_path}")
        return 0

    if args.command == "validate-dataset":
        ok, errors = validate_dataset(args.dataset_dir)
        if ok:
            print("Dataset validation passed.")
            return 0
        print("Dataset validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    if args.command == "review-real-run":
        review_path = review_real_run(args.run_dir)
        print(f"Review written: {review_path}")
        preview_lines = review_path.read_text(encoding="utf-8").splitlines()[:14]
        print("\n".join(preview_lines))
        return 0

    if args.command == "env-check":
        result = probe_environment()
        print(format_environment_summary(result))
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
            print(f"Environment probe written: {out_path}")
        return 0

    if args.command == "debrief-gpu-run":
        debrief_path = debrief_gpu_run(args.run_dir)
        print(f"GPU debrief written: {debrief_path}")
        preview_lines = debrief_path.read_text(encoding="utf-8").splitlines()[:18]
        print("\n".join(preview_lines))
        return 0

    if args.command == "template-report":
        report_path = write_template_autotune_report(
            args.run_dir,
            compare_run_dir=args.compare_run_dir,
        )
        print(f"Template autotune report written: {report_path}")
        preview_lines = report_path.read_text(encoding="utf-8").splitlines()[:18]
        print("\n".join(preview_lines))
        return 0

    if args.command == "profiler-lite":
        report_path = write_profiler_lite_report(args.run_dir)
        print(f"Profiler-lite report written: {report_path}")
        preview_lines = report_path.read_text(encoding="utf-8").splitlines()[:18]
        print("\n".join(preview_lines))
        return 0

    if args.command == "template-copy-report":
        report_path = write_template_copy_report(args.run_dir)
        print(f"Template-copy report written: {report_path}")
        preview_lines = report_path.read_text(encoding="utf-8").splitlines()[:18]
        print("\n".join(preview_lines))
        return 0

    if args.command == "focused-sweep-report":
        report_path = write_focused_sweep_report(
            args.run_dir,
            shapeaware_run=args.shapeaware_run_dir,
            template_copy_wide_run=args.template_copy_wide_run_dir,
        )
        print(f"Focused sweep report written: {report_path}")
        preview_lines = report_path.read_text(encoding="utf-8").splitlines()[:18]
        print("\n".join(preview_lines))
        return 0

    if args.command == "focused-seed-analysis":
        report_path = write_focused_sweep_seed_analysis(
            shapeaware_run=args.shapeaware_run_dir,
            template_copy_wide_run=args.template_copy_wide_run_dir,
            out_path=args.out,
        )
        print(f"Focused seed analysis written: {report_path}")
        preview_lines = report_path.read_text(encoding="utf-8").splitlines()[:18]
        print("\n".join(preview_lines))
        return 0

    if args.command == "repeatability-report":
        report_path, json_path = write_repeatability_report(
            args.run_dir,
            top_k=args.top_k,
            repeats=args.repeats,
        )
        print(f"Repeatability report written: {report_path}")
        print(f"Repeatability JSON written: {json_path}")
        preview_lines = report_path.read_text(encoding="utf-8").splitlines()[:18]
        print("\n".join(preview_lines))
        return 0

    if args.command == "final-3task-report":
        report_path = write_final_3task_report(
            base_template=args.base_template,
            shapeaware=args.shapeaware,
            template_copy_wide=args.template_copy_wide,
            focused=args.focused,
            clean_focused=args.clean_focused,
            out=args.out,
        )
        print(f"Final 3-task report written: {report_path}")
        preview_lines = report_path.read_text(encoding="utf-8").splitlines()[:24]
        print("\n".join(preview_lines))
        return 0

    if args.command == "fused8-report":
        report_path = write_fused8_report(args.run_dir)
        print(f"Fused8 report written: {report_path}")
        preview_lines = report_path.read_text(encoding="utf-8").splitlines()[:24]
        print("\n".join(preview_lines))
        return 0

    if args.command == "curate-fused8-dataset":
        out_path = curate_fused8_dataset(
            template_run=args.template_run,
            gemini_run=args.gemini_run,
            template_guided_run=args.template_guided_run,
            out_dir=args.out_dir,
        )
        print(f"Curated fused8 dataset written: {out_path}")
        print("Repeatability comparison written: runs/fused8_repeatability_comparison.md")
        print("Fused8 conclusion written: runs/fused8_phase11_conclusion.md")
        manifest_path = out_path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            print(json.dumps(manifest.get("counts_by_file", {}), sort_keys=True))
        return 0

    if args.command == "inspect-curated-fused8":
        report_path = inspect_curated_fused8_dataset(args.dataset_dir)
        print(f"Curated fused8 inspection written: {report_path}")
        preview_lines = report_path.read_text(encoding="utf-8").splitlines()[:24]
        print("\n".join(preview_lines))
        return 0

    if args.command == "validate-curated-fused8":
        ok, report_path, errors = validate_curated_fused8_dataset(args.dataset_dir)
        print(f"Curated fused8 validation written: {report_path}")
        print("Curated fused8 validation passed." if ok else "Curated fused8 validation failed.")
        for error in errors[:20]:
            print(f"- {error}")
        return 0 if ok else 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
