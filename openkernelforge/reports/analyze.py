"""Run analysis report generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openkernelforge.reports.failure_taxonomy import classify_candidate_record
from openkernelforge.reports.run_data import load_run_bundle
from openkernelforge.utils.env_probe import environment_warning_for_triton_run


def analyze_run(run_dir: str | Path) -> Path:
    """Create ``analysis.md`` for a run and return its path."""

    bundle = load_run_bundle(run_dir)
    analysis = format_analysis(bundle)
    path = Path(run_dir) / "analysis.md"
    path.write_text(analysis, encoding="utf-8")
    return path


def format_analysis(bundle: dict[str, Any]) -> str:
    run_dir = bundle["run_dir"]
    candidates = bundle["candidate_records"]
    tasks = bundle["task_records"]
    metadata = bundle["metadata"]
    environment = bundle.get("environment") or {}
    config = bundle["config"]
    classifications = [classify_candidate_record(record) for record in candidates]
    taxonomy_counts = Counter(item.failure_type for item in classifications)
    backend = _first(candidates, "backend") or _deep(config, "agent", "backend") or "n/a"
    model = _first(candidates, "model") or _deep(config, "agent", "model") or "n/a"
    agent_type = _first(candidates, "agent_type") or _deep(config, "agent", "type") or "n/a"
    prompt_version = _first(candidates, "prompt_version") or _deep(config, "agent", "prompt_version") or "n/a"
    repair_prompt_version = (
        _first(candidates, "repair_prompt_version")
        or _deep(config, "agent", "repair_prompt_version")
        or "n/a"
    )
    performance_prompt_version = (
        _first(candidates, "performance_prompt_version")
        or _deep(config, "agent", "performance_prompt_version")
        or "n/a"
    )
    harness_only = backend in {"fake", "torch"} or agent_type == "dummy"

    policy_passed = sum(1 for record in candidates if record.get("policy_passed"))
    verified = sum(1 for record in candidates if record.get("verification_passed"))
    benchmarked = sum(1 for record in candidates if record.get("benchmark_summary"))
    selected = [record for record in candidates if record.get("selected_best")]
    selected_best = sum(1 for record in selected)
    performance_candidates = [
        record for record in candidates if record.get("generation_stage") == "performance_search"
    ]
    template_copy_candidates = [
        record for record in candidates if record.get("generation_stage") == "template_copy"
    ]
    target_reached_tasks = len({record.get("task_id") for record in candidates if record.get("target_reached")})

    lines = [
        "# OpenKernelForge Run Analysis",
        "",
        "## Run Metadata",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Backend/model: `{backend}` / `{model}`",
        f"- Agent type: `{agent_type}`",
        f"- Prompt version: `{prompt_version}`",
        f"- Repair prompt version: `{repair_prompt_version}`",
        f"- Performance prompt version: `{performance_prompt_version}`",
        f"- Started: {metadata.get('started_at', 'n/a')}",
        f"- Completed: {metadata.get('completed_at', 'n/a')}",
        f"- Duration seconds: {metadata.get('duration_s', 'n/a')}",
        f"- Harness-only data: {'yes' if harness_only else 'no'}",
    ]
    if environment:
        lines.extend(
            [
                f"- Environment viability: {environment.get('viability', 'n/a')}",
                f"- CUDA available: {'yes' if environment.get('cuda_available') else 'no'}",
                f"- Triton available: {'yes' if environment.get('triton_available') else 'no'}",
                f"- Tiny Triton kernel passed: {'yes' if environment.get('tiny_triton_kernel_passed') else 'no'}",
            ]
        )
        warning = environment_warning_for_triton_run(
            environment,
            requires_triton_kernels=_requires_triton_kernels(config, candidates),
        )
        if warning:
            lines.append(f"- Environment warning: {warning}")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Tasks: {len(tasks)}",
            f"- Candidates: {len(candidates)}",
            f"- Policy passed/failed: {policy_passed}/{len(candidates) - policy_passed}",
            f"- Verification passed/failed: {verified}/{len(candidates) - verified}",
            f"- Benchmarked: {benchmarked}",
            f"- Selected best: {selected_best}",
            f"- Performance-search candidates: {len(performance_candidates)}",
            f"- Template-copy candidates: {len(template_copy_candidates)}",
            f"- Performance-search target reached tasks: {target_reached_tasks}",
            "",
            "## Failure Taxonomy",
            "",
            "| Failure type | Count |",
            "| --- | ---: |",
        ]
    )
    for failure_type, count in sorted(taxonomy_counts.items()):
        lines.append(f"| {failure_type} | {count} |")

    lines.extend(["", "## Failure Categories", ""])
    for name, count in _classification_groups(classifications).items():
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Per-Task Best Candidate", ""])
    if selected:
        lines.extend(["| Task | Candidate | Correct | Speedup vs eager | Failure type |", "| --- | --- | --- | ---: | --- |"])
        for record in selected:
            classification = classify_candidate_record(record)
            benchmark = record.get("benchmark_summary") or {}
            lines.append(
                "| {task} | {candidate} | {correct} | {speedup} | {failure} |".format(
                    task=record.get("task_id"),
                    candidate=record.get("candidate_id"),
                    correct="yes" if record.get("verification_passed") else "no",
                    speedup=_fmt(benchmark.get("speedup_vs_eager")),
                    failure=classification.failure_type,
                )
            )
    else:
        lines.append("No selected candidates were recorded.")

    slow = [
        record
        for record, classification in zip(candidates, classifications, strict=False)
        if classification.failure_type == "CORRECT_BUT_SLOW"
    ]
    lines.extend(["", "## Correct-But-Slow Candidates", ""])
    if slow:
        for record in slow[:10]:
            benchmark = record.get("benchmark_summary") or {}
            lines.append(
                f"- {record.get('task_id')} {record.get('candidate_id')}: "
                f"speedup_vs_eager={benchmark.get('speedup_vs_eager')}"
            )
    else:
        lines.append("None detected.")

    common_errors = Counter(
        (record.get("verification_summary") or {}).get("first_error_type")
        or record.get("policy_rejection_reason")
        or record.get("failure_reason")
        or "none"
        for record in candidates
        if not record.get("verification_passed")
    )
    lines.extend(["", "## Most Common Error Types", ""])
    if common_errors:
        for error, count in common_errors.most_common(10):
            lines.append(f"- {error}: {count}")
    else:
        lines.append("No failed candidates.")

    lines.extend(["", "## Useful Repair Data Examples", ""])
    repair_examples = _repair_examples(candidates)
    if repair_examples:
        for task_id, failed, fixed in repair_examples[:5]:
            lines.append(f"- {task_id}: {failed.get('candidate_id')} -> {fixed.get('candidate_id')}")
    else:
        lines.append("No failed-to-correct sequences found.")

    lines.extend(["", "## Useful SFT Data Examples", ""])
    sft = [record for record in candidates if record.get("policy_passed") and record.get("verification_passed")]
    if sft:
        for record in sft[:5]:
            lines.append(f"- {record.get('task_id')} {record.get('candidate_id')}")
    else:
        lines.append("No policy-passing verified candidates.")

    lines.extend(["", "## Prompt Improvement Recommendations", ""])
    lines.extend(_recommendations(candidates, harness_only))

    lines.extend(["", "## Dataset Export Usefulness", ""])
    if harness_only:
        lines.append("This run appears to be dummy/fake harness data. Use it to test export plumbing, not for model training.")
    elif verified:
        lines.append("This run has verified real-model candidates and is worth manual review before dataset export.")
    else:
        lines.append("This run has no verified candidates; export may still be useful for rejected/failure analysis.")
    lines.append("")
    return "\n".join(lines)


def _repair_examples(candidates: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        by_task[str(record.get("task_id"))].append(record)
    pairs = []
    for task_id, records in by_task.items():
        records = sorted(records, key=lambda r: (r.get("attempt_index") or 0, r.get("candidate_index") or 0))
        failed = [record for record in records if not record.get("verification_passed")]
        correct = [record for record in records if record.get("verification_passed")]
        if failed and correct:
            pairs.append((task_id, failed[0], correct[0]))
    return pairs


def _recommendations(candidates: list[dict[str, Any]], harness_only: bool) -> list[str]:
    recommendations = []
    policy_rejections = sum(1 for record in candidates if not record.get("policy_passed"))
    extraction_failures = sum(1 for record in candidates if record.get("failure_reason") == "code_extraction_failed")
    mismatches = sum(
        1
        for record in candidates
        if (record.get("verification_summary") or {}).get("first_error_type") == "values_not_close"
    )
    if harness_only:
        recommendations.append("- Harness-only run: use a real backend before judging prompt quality.")
    if policy_rejections:
        recommendations.append("- Many candidates hit policy rejection; emphasize Triton implementation requirements and forbid direct PyTorch fallback.")
    if extraction_failures:
        recommendations.append("- Extraction failed; ask the model to return only Python code with a top-level `forward` function.")
    if mismatches:
        recommendations.append("- Numerical mismatches found; include tolerance and shape details more prominently in repair prompts.")
    if not recommendations:
        recommendations.append("- No obvious prompt issue detected from this run.")
    return recommendations


def _classification_groups(classifications: list[Any]) -> dict[str, int]:
    groups = {
        "model generation failures": 0,
        "policy failures": 0,
        "local execution environment failures": 0,
        "correctness failures": 0,
        "benchmark failures": 0,
    }
    for classification in classifications:
        failure_type = classification.failure_type
        if failure_type.startswith("POLICY_"):
            groups["policy failures"] += 1
        elif failure_type.startswith("ENV_"):
            groups["local execution environment failures"] += 1
        elif failure_type in {"EXTRACTION_FAILURE", "SYNTAX_ERROR", "IMPORT_ERROR", "MODEL_IMPORT_ERROR"}:
            groups["model generation failures"] += 1
        elif failure_type in {
            "TRITON_COMPILE_ERROR",
            "RUNTIME_ERROR",
            "SHAPE_MISMATCH",
            "DTYPE_MISMATCH",
            "NUMERICAL_MISMATCH",
            "NAN_OR_INF",
        }:
            groups["correctness failures"] += 1
        elif failure_type == "BENCHMARK_ERROR":
            groups["benchmark failures"] += 1
    return groups


def _requires_triton_kernels(config: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    agent = config.get("agent") or {}
    execution = config.get("execution") or {}
    return bool(
        execution.get("require_triton")
        or not agent.get("allow_torch_fallback", True)
        or any(record.get("backend") == "openai_compatible" for record in candidates)
    )


def _first(records: list[dict[str, Any]], key: str) -> Any:
    for record in records:
        value = record.get(key)
        if value:
            return value
    return None


def _deep(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
