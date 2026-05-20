"""Summarize OpenKernelForge run results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_results(run_dir: str | Path) -> list[dict[str, Any]]:
    results_path = Path(run_dir) / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")
    records: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_summary(run_dir: str | Path, records: list[dict[str, Any]] | None = None) -> Path:
    run_path = Path(run_dir)
    selected_records = records if records is not None else load_results(run_path)
    environment = _load_environment_probe(run_path)
    summary_path = run_path / "summary.md"
    summary_path.write_text(
        format_summary(selected_records, environment=environment),
        encoding="utf-8",
    )
    return summary_path


def format_summary(
    records: list[dict[str, Any]],
    *,
    environment: dict[str, Any] | None = None,
) -> str:
    task_records, candidate_records = _split_records(records)
    total = len(task_records)
    verified = sum(1 for record in task_records if record.get("verification", {}).get("passed"))
    policy_passed = sum(1 for record in candidate_records if record.get("policy_passed"))
    verification_passed = sum(1 for record in candidate_records if record.get("verification_passed"))
    benchmarked = sum(1 for record in candidate_records if record.get("benchmark_summary"))
    backend = _first_nonempty(candidate_records, "backend") or _first_nonempty(task_records, "backend")
    model = _first_nonempty(candidate_records, "model")
    prompt_version = _first_nonempty(candidate_records, "prompt_version")
    repair_prompt_version = _first_nonempty(candidate_records, "repair_prompt_version")
    performance_prompt_version = _first_nonempty(candidate_records, "performance_prompt_version")
    performance_candidates = [
        record for record in candidate_records if record.get("generation_stage") == "performance_search"
    ]
    template_copy_candidates = [
        record for record in candidate_records if record.get("generation_stage") == "template_copy"
    ]
    target_reached_tasks = len(
        {
            record.get("task_id")
            for record in candidate_records
            if record.get("target_reached")
        }
    )
    lines = [
        "# OpenKernelForge Run Summary",
        "",
        f"- Tasks: {total}",
        f"- Backend: {backend or 'n/a'}",
        f"- Model: {model or 'n/a'}",
        f"- Prompt version: {prompt_version or 'n/a'}",
        f"- Repair prompt version: {repair_prompt_version or 'n/a'}",
        f"- Performance prompt version: {performance_prompt_version or 'n/a'}",
        f"- Candidates: {len(candidate_records)}",
        f"- Performance-search candidates: {len(performance_candidates)}",
        f"- Template-copy candidates: {len(template_copy_candidates)}",
        f"- Policy passed/failed: {policy_passed}/{len(candidate_records) - policy_passed}",
        f"- Verification passed/failed: {verification_passed}/{len(candidate_records) - verification_passed}",
        f"- Benchmarked candidates: {benchmarked}",
        f"- Selected correct tasks: {verified}",
        f"- Performance-search target reached tasks: {target_reached_tasks}",
    ]
    if environment:
        lines.extend(
            [
                f"- Environment viability: {environment.get('viability', 'n/a')}",
                f"- CUDA available: {'yes' if environment.get('cuda_available') else 'no'}",
                f"- Triton available: {'yes' if environment.get('triton_available') else 'no'}",
            ]
        )
        warning = _environment_warning(environment, candidate_records)
        if warning:
            lines.append(f"- Environment warning: {warning}")
    lines.extend(
        [
            "",
            "## Task Results",
            "",
            "| Task | Agent | Attempts | Candidates | Candidate | Verified | Eager median ms | Candidate median ms | Speedup vs eager | Notes |",
            "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for record in task_records:
        verification = record.get("verification", {})
        benchmarks = record.get("benchmarks") or []
        benchmark = benchmarks[0] if benchmarks else {}
        eager = (benchmark.get("eager") or {}).get("median_ms")
        candidate = (benchmark.get("candidate") or {}).get("median_ms")
        speedup = benchmark.get("speedup_vs_eager")
        notes = _record_notes(record)
        lines.append(
            "| {task} | {agent} | {attempts} | {candidates} | {candidate_name} | {verified} | {eager} | {candidate_ms} | {speedup} | {notes} |".format(
                task=record.get("task_id", "unknown"),
                agent=record.get("agent_type", "unknown"),
                attempts=_attempt_count(record),
                candidates=len(record.get("attempts") or []) or 1,
                candidate_name=record.get("candidate_name", "unknown"),
                verified="yes" if verification.get("passed") else "no",
                eager=_fmt_ms(eager),
                candidate_ms=_fmt_ms(candidate),
                speedup=_fmt_speedup(speedup),
                notes=notes,
            )
        )
    lines.extend(["", "## Selected Speedups", ""])
    selected = [record for record in candidate_records if record.get("selected_best")]
    if selected:
        lines.extend(
            [
                "| Task | Candidate | Policy | Correct | Speedup vs eager | Speedup vs torch.compile | Candidate median ms |",
                "| --- | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for record in selected:
            benchmark = record.get("benchmark_summary") or {}
            lines.append(
                "| {task} | {candidate} | {policy} | {correct} | {eager} | {compile} | {median} |".format(
                    task=record.get("task_id", "unknown"),
                    candidate=record.get("candidate_id", "unknown"),
                    policy="yes" if record.get("policy_passed") else "no",
                    correct="yes" if record.get("verification_passed") else "no",
                    eager=_fmt_speedup(benchmark.get("speedup_vs_eager")),
                    compile=_fmt_speedup(benchmark.get("speedup_vs_torch_compile")),
                    median=_fmt_ms(benchmark.get("candidate_median_ms")),
                )
            )
    else:
        lines.append("No selected candidates were recorded.")

    lines.extend(["", "## Failure Groups", ""])
    failure_groups = _failure_groups(candidate_records)
    for name, items in failure_groups.items():
        lines.append(f"- {name}: {len(items)}")
        for item in items[:5]:
            detail = item.get("policy_rejection_reason") or item.get("failure_reason") or "unknown"
            lines.append(f"  - {item.get('task_id')} {item.get('candidate_id')}: {detail}")
    lines.append("")
    return "\n".join(lines)


def _split_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_records = [
        record for record in records if record.get("record_type", "task_summary") != "candidate"
    ]
    top_level_candidates = [record for record in records if record.get("record_type") == "candidate"]
    if top_level_candidates:
        return task_records, top_level_candidates
    nested_candidates = [
        candidate
        for record in task_records
        for candidate in record.get("candidate_records", [])
    ]
    return task_records, nested_candidates


def _record_notes(record: dict[str, Any]) -> str:
    verification = record.get("verification", {})
    if verification.get("passed"):
        benchmarks = record.get("benchmarks") or []
        if benchmarks:
            bench_error = benchmarks[0].get("benchmark_error")
            if bench_error:
                return "benchmark error"
        return ""
    cases = verification.get("cases") or []
    for case in cases:
        if not case.get("passed"):
            return str(case.get("error_type") or "verification failed")
    attempts = record.get("attempts") or []
    if attempts:
        failure = attempts[-1].get("failure_reason")
        if failure:
            return str(failure)
    if verification.get("error"):
        return "candidate load or verification error"
    return "not verified"


def _failure_groups(candidate_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "policy rejection": [],
        "extraction failure": [],
        "compile/runtime error": [],
        "correctness mismatch": [],
        "benchmark failure": [],
    }
    for record in candidate_records:
        reason = record.get("failure_reason")
        benchmark = record.get("benchmark_summary") or {}
        verification = record.get("verification_summary") or {}
        if not record.get("policy_passed"):
            groups["policy rejection"].append(record)
        if reason == "code_extraction_failed":
            groups["extraction failure"].append(record)
        if reason == "exception" or verification.get("first_error_type") == "exception":
            groups["compile/runtime error"].append(record)
        if reason in {"values_not_close", "wrong_shape", "wrong_dtype", "nonfinite_output"}:
            groups["correctness mismatch"].append(record)
        if benchmark.get("benchmark_error") or benchmark.get("compile_error"):
            groups["benchmark failure"].append(record)
    return groups


def _first_nonempty(records: list[dict[str, Any]], key: str) -> Any:
    for record in records:
        value = record.get(key)
        if value:
            return value
    return None


def _attempt_count(record: dict[str, Any]) -> int:
    attempts = record.get("attempts") or []
    if not attempts:
        return 1
    return len({attempt.get("attempt_index") for attempt in attempts})


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _fmt_speedup(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}x"


def _load_environment_probe(run_path: Path) -> dict[str, Any] | None:
    path = run_path / "environment_probe.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _environment_warning(
    environment: dict[str, Any],
    candidate_records: list[dict[str, Any]],
) -> str | None:
    requires_triton = any(
        record.get("backend") == "openai_compatible"
        or "triton" in str(record.get("policy_warnings") or "").lower()
        for record in candidate_records
    )
    if not requires_triton:
        return None
    if environment.get("viability") in {
        "CPU_ONLY",
        "CUDA_NO_TRITON",
        "TRITON_IMPORT_ONLY",
        "UNKNOWN_BROKEN",
    }:
        return (
            "This run can test model generation and policy behavior, but cannot "
            "verify/benchmark Triton kernels on this machine."
        )
    return None
