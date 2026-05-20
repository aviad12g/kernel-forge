"""Human-oriented review report for real-model baseline runs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from openkernelforge.reports.failure_taxonomy import classify_candidate_record
from openkernelforge.reports.run_data import load_run_bundle
from openkernelforge.utils.env_probe import environment_warning_for_triton_run


def review_real_run(run_dir: str | Path) -> Path:
    """Create ``real_run_review.md`` for a run and return its path."""

    bundle = load_run_bundle(run_dir)
    text = format_real_run_review(bundle)
    path = Path(run_dir) / "real_run_review.md"
    path.write_text(text, encoding="utf-8")
    return path


def format_real_run_review(bundle: dict[str, Any]) -> str:
    run_dir = bundle["run_dir"]
    candidates = bundle["candidate_records"]
    tasks = bundle["task_records"]
    config = bundle["config"]
    environment = bundle.get("environment") or {}
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
    template_run = agent_type == "template" or backend == "triton_template"
    harness_only = backend in {"fake", "torch"} or agent_type == "dummy"
    classifications = {id(record): classify_candidate_record(record) for record in candidates}
    behavior = _behavior_counts(candidates, classifications)

    lines = [
        "# OpenKernelForge Real Run Review",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Appears real model run: {_real_model_label(harness_only, template_run)}",
        f"- Agent/backend/model: `{agent_type}` / `{backend}` / `{model}`",
        f"- Prompt version: `{prompt_version}`",
        f"- Repair prompt version: `{repair_prompt_version}`",
        f"- Performance prompt version: `{performance_prompt_version}`",
        f"- Candidates: {len(candidates)}",
        f"- Performance-search candidates: {sum(1 for record in candidates if record.get('generation_stage') == 'performance_search')}",
    ]
    if environment:
        lines.extend(
            [
                f"- Environment viability: `{environment.get('viability', 'n/a')}`",
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
            "## Per-Task Outcome",
            "",
            "| Task | Best candidate | Selected | Policy | Verifier | Benchmark | Failure type |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for task in tasks:
        task_id = task.get("task_id")
        task_candidates = [record for record in candidates if record.get("task_id") == task_id]
        selected = next((record for record in task_candidates if record.get("selected_best")), None)
        best = selected or (task_candidates[-1] if task_candidates else {})
        classification = classify_candidate_record(best) if best else None
        benchmark = best.get("benchmark_summary") or {}
        lines.append(
            "| {task} | {candidate} | {selected} | {policy} | {verifier} | {benchmark} | {failure} |".format(
                task=task_id,
                candidate=best.get("candidate_id", "n/a"),
                selected="yes" if best.get("selected_best") else "no",
                policy="pass" if best.get("policy_passed") else best.get("policy_rejection_reason", "fail"),
                verifier="pass" if best.get("verification_passed") else _verifier_note(best),
                benchmark=_benchmark_note(benchmark),
                failure=classification.failure_type if classification else "n/a",
            )
        )

    lines.extend(
        [
            "",
            "## Common Model Behavior",
            "",
            f"- Returns fallback PyTorch: {behavior['fallback']}",
            f"- Bad code extraction: {behavior['extraction']}",
            f"- Syntax errors: {behavior['syntax']}",
            f"- Triton compile errors: {behavior['triton_compile']}",
            f"- Local execution environment failures: {behavior['environment']}",
            f"- Numerical mismatch: {behavior['numerical']}",
            f"- Correct but slow: {behavior['correct_slow']}",
            "",
            "## Prompt Weaknesses Detected",
            "",
        ]
    )
    lines.extend(_prompt_weaknesses(behavior, harness_only))
    lines.extend(["", "## Recommended Next Prompt Changes", ""])
    lines.extend(_prompt_recommendations(behavior, harness_only))

    sft_count = sum(1 for record in candidates if record.get("policy_passed") and record.get("verification_passed"))
    repair_possible = _repair_possible(candidates)
    optimization_possible = _optimization_possible(candidates)
    lines.extend(
        [
            "",
            "## Dataset Usefulness",
            "",
            f"- SFT: {'candidate rows available' if sft_count else 'no verified policy-passing candidates'} ({sft_count})",
            f"- Repair training: {'candidate pairs available' if repair_possible else 'no failed-to-fixed sequence detected'}",
            f"- Optimization training: {'candidate pairs available' if optimization_possible else 'no benchmark-supported faster/slower correct pair detected'}",
            "",
            "## Human Review Checklist",
            "",
            "- Confirm this is a real model run, not fake/dummy harness output.",
            "- Inspect prompts and raw responses for leakage, shortcuts, and fallback behavior.",
            "- Inspect selected candidate code before using it as training target.",
            "- Confirm correctness tolerances and benchmark shapes are appropriate.",
            "- Treat correct-but-slow candidates as optimization data, not high-quality SFT targets.",
            "- Do not fine-tune until dataset rows are manually reviewed.",
            "",
        ]
    )
    return "\n".join(lines)


def _behavior_counts(candidates: list[dict[str, Any]], classifications: dict[int, Any]) -> Counter:
    counts = Counter()
    for record in candidates:
        classification = classifications[id(record)]
        failure = classification.failure_type
        if failure == "POLICY_REJECTED_TORCH_FALLBACK":
            counts["fallback"] += 1
        if failure == "EXTRACTION_FAILURE":
            counts["extraction"] += 1
        if failure == "SYNTAX_ERROR":
            counts["syntax"] += 1
        if failure == "TRITON_COMPILE_ERROR":
            counts["triton_compile"] += 1
        if failure.startswith("ENV_"):
            counts["environment"] += 1
        if failure == "NUMERICAL_MISMATCH":
            counts["numerical"] += 1
        if failure == "CORRECT_BUT_SLOW":
            counts["correct_slow"] += 1
    return counts


def _real_model_label(harness_only: bool, template_run: bool) -> str:
    if template_run:
        return "no (deterministic template baseline)"
    if harness_only:
        return "no (harness-only)"
    return "yes"


def _prompt_weaknesses(behavior: Counter, harness_only: bool) -> list[str]:
    lines = []
    if harness_only:
        lines.append("- Harness-only run: prompt weaknesses from this report are not real-model evidence.")
    if behavior["fallback"]:
        lines.append("- Model still returns PyTorch fallback despite no-fallback policy.")
    if behavior["environment"]:
        lines.append("- Local execution environment prevented Triton verification or benchmarking.")
    if behavior["extraction"] or behavior["syntax"]:
        lines.append("- Model may need stronger instruction to return only complete Python code.")
    if behavior["numerical"]:
        lines.append("- Prompt may need clearer shape, broadcasting, or tolerance details.")
    if not lines:
        lines.append("- No obvious prompt weakness detected from available records.")
    return lines


def _prompt_recommendations(behavior: Counter, harness_only: bool) -> list[str]:
    lines = []
    if behavior["fallback"]:
        lines.append("- Add a short Triton skeleton example for the task family and repeat that PyTorch fallback is rejected.")
    if behavior["triton_compile"]:
        lines.append("- Ask for simpler block sizes and fewer advanced Triton features.")
    if behavior["environment"]:
        lines.append("- Rerun on a CUDA machine with Triton installed before treating import/runtime failures as model failures.")
    if behavior["numerical"]:
        lines.append("- Include explicit indexing formulas and broadcasting semantics in task-specific hints.")
    if behavior["correct_slow"]:
        lines.append("- Add optimization feedback after correctness is established.")
    if harness_only:
        lines.append("- Run against a real backend before making prompt changes based on this review.")
    if not lines:
        lines.append("- Keep current prompt for the next small real-model run and gather more evidence.")
    return lines


def _repair_possible(candidates: list[dict[str, Any]]) -> bool:
    seen_failed: set[str] = set()
    for record in sorted(candidates, key=lambda r: (r.get("task_id"), r.get("attempt_index") or 0, r.get("candidate_index") or 0)):
        task_id = str(record.get("task_id"))
        if not record.get("verification_passed"):
            seen_failed.add(task_id)
        elif task_id in seen_failed:
            return True
    return False


def _optimization_possible(candidates: list[dict[str, Any]]) -> bool:
    by_task: dict[str, list[float]] = {}
    for record in candidates:
        if not record.get("verification_passed"):
            continue
        benchmark = record.get("benchmark_summary") or {}
        median = benchmark.get("candidate_median_ms")
        if median is not None:
            by_task.setdefault(str(record.get("task_id")), []).append(float(median))
    return any(len(values) >= 2 and max(values) > min(values) * 1.05 for values in by_task.values())


def _verifier_note(record: dict[str, Any]) -> str:
    summary = record.get("verification_summary") or {}
    return summary.get("first_error_type") or record.get("failure_reason") or "fail"


def _benchmark_note(benchmark: dict[str, Any]) -> str:
    if not benchmark:
        return "n/a"
    if benchmark.get("benchmark_error"):
        return "benchmark error"
    speedup = benchmark.get("speedup_vs_eager")
    return f"{float(speedup):.3f}x eager" if speedup is not None else "benchmarked"


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


def _requires_triton_kernels(config: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    agent = config.get("agent") or {}
    execution = config.get("execution") or {}
    return bool(
        execution.get("require_triton")
        or not agent.get("allow_torch_fallback", True)
        or any(record.get("backend") == "openai_compatible" for record in candidates)
    )
