"""Benchmark/statics profiler-lite reporting.

This is intentionally not Nsight or a hardware profiler. It combines saved
benchmark summaries with static source heuristics to guide prompt changes.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from openkernelforge.reports.run_data import load_run_bundle, read_artifact


def write_profiler_lite_report(run_dir: str | Path) -> Path:
    """Write ``profiler_lite_report.md`` for a run."""

    bundle = load_run_bundle(run_dir)
    path = Path(run_dir) / "profiler_lite_report.md"
    path.write_text(format_profiler_lite_report(bundle), encoding="utf-8")
    return path


def format_profiler_lite_report(bundle: dict[str, Any]) -> str:
    run_dir = bundle["run_dir"]
    candidates = bundle["candidate_records"]
    by_task = _by_task(candidates)
    lines = [
        "# OpenKernelForge Profiler-Lite Report",
        "",
        f"- Run dir: `{run_dir}`",
        "- Method: saved benchmark statistics plus static source heuristics.",
        "- Important: these are static heuristics, not hardware-profiler facts.",
        "",
        "## Fastest Candidates",
        "",
        "| Task | Fastest candidate | Stage | Speedup vs eager | Median ms | Source |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for task_id, records in sorted(by_task.items()):
        best = _best_record(records)
        benchmark = best.get("benchmark_summary") or {}
        lines.append(
            "| {task} | {candidate} | {stage} | {speedup} | {median} | `{source}` |".format(
                task=task_id,
                candidate=best.get("candidate_id", "n/a"),
                stage=best.get("generation_stage", "n/a"),
                speedup=_fmt_speedup(benchmark.get("speedup_vs_eager")),
                median=_fmt_ms(benchmark.get("candidate_median_ms")),
                source=best.get("candidate_path", "n/a"),
            )
        )

    lines.extend(["", "## Runtime Distribution By Task", ""])
    for task_id, records in sorted(by_task.items()):
        speedups = _speedups(records)
        medians = _candidate_medians(records)
        best_speedup = max(speedups) if speedups else None
        med = median(speedups) if speedups else None
        spread = (max(medians) - min(medians)) if len(medians) >= 2 else None
        lines.extend(
            [
                f"### {task_id}",
                "",
                f"- Candidate count: {len(records)}",
                f"- Best speedup vs eager: {_fmt_speedup(best_speedup)}",
                f"- Median speedup vs eager: {_fmt_speedup(med)}",
                f"- Runtime spread between slowest and fastest correct candidate: {_fmt_ms(spread)}",
                "",
                "| Candidate | Stage | Median ms | Mean ms | p25 ms | p75 ms | Speedup eager | Speedup compile | Static flags |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for record in sorted(records, key=_speedup_key, reverse=True):
            stats = _candidate_stats(record)
            flags = _static_flags(read_artifact(record.get("candidate_path"), run_dir=run_dir), str(task_id))
            concise_flags = ", ".join(_flag_names(flags)) or "none"
            benchmark = record.get("benchmark_summary") or {}
            lines.append(
                "| {candidate} | {stage} | {median} | {mean} | {p25} | {p75} | {eager} | {compile} | {flags} |".format(
                    candidate=record.get("candidate_id", "n/a"),
                    stage=record.get("generation_stage", "n/a"),
                    median=_fmt_ms(stats.get("median_ms")),
                    mean=_fmt_ms(stats.get("mean_ms")),
                    p25=_fmt_ms(stats.get("p25_ms")),
                    p75=_fmt_ms(stats.get("p75_ms")),
                    eager=_fmt_speedup(benchmark.get("speedup_vs_eager")),
                    compile=_fmt_speedup(benchmark.get("speedup_vs_torch_compile")),
                    flags=concise_flags,
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Static Pattern Correlations",
            "",
            "| Pattern | Present count | Avg speedup when present | Avg speedup when absent |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    correlations = _pattern_correlations(candidates, run_dir)
    for name, data in sorted(correlations.items()):
        lines.append(
            f"| {name} | {data['present_count']} | {_fmt_speedup(data['present_avg'])} | "
            f"{_fmt_speedup(data['absent_avg'])} |"
        )

    lines.extend(
        [
            "",
            "## Recommendations For Template-Copy Prompt",
            "",
            "- Preserve the fastest template wrapper and kernel launch structure exactly.",
            "- Reject added try/except fallback branches before verification.",
            "- Reject forbidden torch ops such as torch.relu, torch.maximum, torch.add, torch.matmul, and torch.sigmoid in forward.",
            "- Keep exactly one Triton launch for these elementwise tasks.",
            "- Keep bias_relu modulo feature indexing when copying bias_relu templates.",
            "",
        ]
    )
    return "\n".join(lines)


def source_static_flags(source: str, task_id: str | None = None) -> dict[str, Any]:
    """Return static source flags used by profiler-lite and comparison reports."""

    forward = _forward_source(source)
    return {
        "extra_torch_ops": bool(_forbidden_torch_ops(forward)),
        "forbidden_torch_ops": _forbidden_torch_ops(forward),
        "contiguous_calls": len(re.findall(r"\.\s*contiguous\s*\(", forward)),
        "uses_empty_like": "torch.empty_like" in forward,
        "uses_empty": "torch.empty(" in forward,
        "python_fallback_branch": _fallback_detected(source),
        "triton_jit_present": "@triton.jit" in source,
        "block_size_constexpr_present": "BLOCK_SIZE" in source and "tl.constexpr" in source,
        "num_warps_specified": "num_warps=" in source,
        "kernel_launch_count": len(re.findall(r"\w+\s*\[\s*grid\s*\]\s*\(", source)),
        "bias_modulo_indexing": "%" in source and (
            "feature_dim" in source or "features" in source or "bias.numel" in source
        ),
        "uses_tl_maximum": "tl.maximum" in source,
        "uses_forbidden_torch_activation": any(
            op in forward for op in ("torch.relu", "torch.maximum", "torch.add")
        ),
        "try_except_import_fallback": "try:" in source and "except" in source,
        "task_id": task_id,
    }


def _by_task(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        grouped[str(record.get("task_id"))].append(record)
    return grouped


def _best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(records, key=_speedup_key) if records else {}


def _speedup_key(record: dict[str, Any]) -> tuple[int, float]:
    speedup = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
    return (1 if record.get("verification_passed") else 0, float(speedup) if speedup is not None else -1.0)


def _candidate_stats(record: dict[str, Any]) -> dict[str, Any]:
    benchmark_result = record.get("benchmark_result") or []
    if benchmark_result:
        candidate = (benchmark_result[0] or {}).get("candidate") or {}
        if candidate:
            return candidate
    summary = record.get("benchmark_summary") or {}
    median_ms = summary.get("candidate_median_ms")
    return {
        "median_ms": median_ms,
        "mean_ms": median_ms,
        "p25_ms": median_ms,
        "p75_ms": median_ms,
    }


def _candidate_medians(records: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _candidate_stats(record).get("median_ms")
        if value is not None:
            values.append(float(value))
    return values


def _speedups(records: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for record in records:
        value = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
        if value is not None:
            values.append(float(value))
    return values


def _static_flags(source: str, task_id: str) -> dict[str, Any]:
    return source_static_flags(source, task_id)


def _flag_names(flags: dict[str, Any]) -> list[str]:
    names: list[str] = []
    if flags.get("extra_torch_ops"):
        names.append("extra_torch_ops")
    if flags.get("contiguous_calls"):
        names.append("contiguous_calls")
    if flags.get("python_fallback_branch"):
        names.append("fallback_branch")
    if not flags.get("triton_jit_present"):
        names.append("missing_triton_jit")
    if not flags.get("block_size_constexpr_present"):
        names.append("missing_block_size_constexpr")
    if not flags.get("num_warps_specified"):
        names.append("num_warps_missing")
    if flags.get("kernel_launch_count") != 1:
        names.append(f"kernel_launches={flags.get('kernel_launch_count')}")
    if flags.get("task_id") == "bias_relu" and not flags.get("bias_modulo_indexing"):
        names.append("bias_indexing_unclear")
    return names


def _pattern_correlations(candidates: list[dict[str, Any]], run_dir: Path) -> dict[str, dict[str, Any]]:
    names = [
        "extra_torch_ops",
        "contiguous_calls",
        "uses_empty_like",
        "python_fallback_branch",
        "triton_jit_present",
        "block_size_constexpr_present",
        "num_warps_specified",
        "bias_modulo_indexing",
        "uses_tl_maximum",
        "try_except_import_fallback",
    ]
    rows: dict[str, dict[str, list[float]]] = {
        name: {"present": [], "absent": []} for name in names
    }
    for record in candidates:
        speedup = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
        if speedup is None:
            continue
        flags = source_static_flags(
            read_artifact(record.get("candidate_path"), run_dir=run_dir),
            str(record.get("task_id")),
        )
        for name in names:
            present = bool(flags.get(name))
            if name == "contiguous_calls":
                present = bool(flags.get(name, 0))
            rows[name]["present" if present else "absent"].append(float(speedup))
    return {
        name: {
            "present_count": len(values["present"]),
            "present_avg": _avg(values["present"]),
            "absent_avg": _avg(values["absent"]),
        }
        for name, values in rows.items()
    }


def _forward_source(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            start = max(node.lineno - 1, 0)
            end = int(getattr(node, "end_lineno", node.lineno))
            return "\n".join(lines[start:end])
    return source


def _forbidden_torch_ops(forward: str) -> list[str]:
    return sorted(set(re.findall(r"torch\.(relu|maximum|add|matmul|sigmoid|mul)\b", forward)))


def _fallback_detected(source: str) -> bool:
    lowered = source.lower()
    return any(token in lowered for token in ("fallback", "ref_forward", "torch_forward")) or (
        "try:" in lowered and "except" in lowered
    )


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _fmt_speedup(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}x"
