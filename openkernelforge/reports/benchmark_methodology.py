"""Sanity checks for rigorous benchmark methodology settings."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openkernelforge.config import RunConfig, load_config, save_config
from openkernelforge.harness.runner import run_from_config
from openkernelforge.utils.env_probe import format_environment_summary, probe_environment


def run_benchmark_methodology_check(
    config_path: str | Path,
    *,
    max_tasks: int = 2,
) -> Path:
    """Run or skip a tiny methodology check and write a Markdown report."""

    config = load_config(config_path)
    environment = probe_environment()
    if not environment.cuda_available:
        run_dir = _make_report_dir(config.output_dir)
        save_config(config, run_dir / "config.yaml")
        (run_dir / "environment_probe.json").write_text(
            json.dumps(environment.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        report_path = run_dir / "benchmark_methodology_check.md"
        report_path.write_text(
            _format_skipped_report(config_path, environment, reason="CUDA is unavailable"),
            encoding="utf-8",
        )
        return report_path

    check_config = _tiny_check_config(config, max_tasks=max_tasks)
    run_dir = run_from_config(check_config)
    report_path = Path(run_dir) / "benchmark_methodology_check.md"
    records = _load_candidate_records(Path(run_dir) / "results.jsonl")
    report_path.write_text(
        _format_completed_report(config_path, environment, run_dir, records),
        encoding="utf-8",
    )
    return report_path


def _tiny_check_config(config: RunConfig, *, max_tasks: int) -> RunConfig:
    data = config.to_dict()
    data["tasks"] = list(config.tasks[: max(1, max_tasks)])
    agent = data.setdefault("agent", {})
    variants = deepcopy(agent.get("template_variants") or {})
    variants["max_variants_per_task"] = min(int(variants.get("max_variants_per_task", 1)), 1)
    variants["grid_sampling"] = "capped_ordered"
    agent["template_variants"] = variants
    return RunConfig.from_dict(data)


def _format_skipped_report(config_path: str | Path, environment: Any, *, reason: str) -> str:
    return "\n".join(
        [
            "# Benchmark Methodology Check",
            "",
            f"- Config: `{config_path}`",
            f"- Status: skipped",
            f"- Reason: {reason}",
            "",
            "## Environment",
            "",
            "```text",
            format_environment_summary(environment),
            "```",
            "",
            "This check requires CUDA for CUDA-event timing. The command exited cleanly without running model or template experiments.",
            "",
        ]
    )


def _format_completed_report(
    config_path: str | Path,
    environment: Any,
    run_dir: str | Path,
    records: list[dict[str, Any]],
) -> str:
    summaries = [record.get("benchmark_summary") or {} for record in records if record.get("benchmark_summary")]
    first = summaries[0] if summaries else {}
    lines = [
        "# Benchmark Methodology Check",
        "",
        f"- Config: `{config_path}`",
        f"- Run dir: `{run_dir}`",
        "- Status: completed",
        f"- Candidate records: {len(records)}",
        f"- Timing mode: {first.get('timing_mode', 'n/a')}",
        f"- Cache flush enabled: {first.get('cache_flush_enabled', 'n/a')}",
        f"- Cache flush performed: {first.get('cache_flush_performed', 'n/a')}",
        f"- Independent sessions: {first.get('independent_sessions', 'n/a')}",
        "",
        "## Environment",
        "",
        "```text",
        format_environment_summary(environment),
        "```",
        "",
        "## First Benchmark Summary",
        "",
        f"- Candidate median: {_fmt((first.get('candidate_ms_summary') or {}).get('median_ms'))} ms",
        f"- Candidate IQR: {_fmt((first.get('candidate_ms_summary') or {}).get('iqr_ms'))} ms",
        f"- Candidate CV: {_fmt((first.get('candidate_ms_summary') or {}).get('cv'))}",
        f"- Eager median: {_fmt((first.get('eager_ms_summary') or {}).get('median_ms'))} ms",
        f"- Speedup vs eager: {_fmt(first.get('speedup_vs_eager'))}x",
        f"- torch.compile median: {_fmt((first.get('torch_compile_ms_summary') or {}).get('median_ms'))} ms",
        f"- torch.compile error: {'yes' if first.get('compile_error') else 'no'}",
        "",
    ]
    warnings = first.get("measurement_warnings") or []
    if warnings:
        lines.extend(["## Measurement Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    return "\n".join(lines)


def _load_candidate_records(results_path: Path) -> list[dict[str, Any]]:
    if not results_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("record_type") == "candidate":
            records.append(record)
    return records


def _make_report_dir(output_dir: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_methodology_check")
    path = Path(output_dir) / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
