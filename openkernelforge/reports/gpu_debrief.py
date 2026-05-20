"""GPU candidate debrief reports for CUDA/Triton baseline runs."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openkernelforge.reports.failure_taxonomy import classify_candidate_record
from openkernelforge.reports.run_data import load_run_bundle, read_artifact


def debrief_gpu_run(run_dir: str | Path) -> Path:
    """Create ``gpu_candidate_debrief.md`` for a GPU run and return its path."""

    bundle = load_run_bundle(run_dir)
    text = format_gpu_debrief(bundle)
    path = Path(run_dir) / "gpu_candidate_debrief.md"
    path.write_text(text, encoding="utf-8")
    return path


def format_gpu_debrief(bundle: dict[str, Any]) -> str:
    run_dir = Path(bundle["run_dir"])
    candidates = bundle["candidate_records"]
    tasks = bundle["task_records"]
    environment = bundle.get("environment") or {}
    config = bundle.get("config") or {}
    classifications = {id(record): classify_candidate_record(record) for record in candidates}
    heuristic_counts: Counter[str] = Counter()

    lines = [
        "# OpenKernelForge GPU Candidate Debrief",
        "",
        "## Environment Summary",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Viability: `{environment.get('viability', 'n/a')}`",
        f"- CUDA available: {'yes' if environment.get('cuda_available') else 'no'}",
        f"- Triton available: {'yes' if environment.get('triton_available') else 'no'}",
        f"- Tiny Triton kernel passed: {'yes' if environment.get('tiny_triton_kernel_passed') else 'no'}",
        f"- Prompt version: `{_deep(config, 'agent', 'prompt_version') or _first(candidates, 'prompt_version') or 'n/a'}`",
        f"- Repair prompt version: `{_deep(config, 'agent', 'repair_prompt_version') or _first(candidates, 'repair_prompt_version') or 'n/a'}`",
        f"- Performance prompt version: `{_deep(config, 'agent', 'performance_prompt_version') or _first(candidates, 'performance_prompt_version') or 'n/a'}`",
        "",
        "## Task-Level Outcome Summary",
        "",
        "| Task | Verified | Selected candidate | Speedup vs eager | Notes |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for task in tasks:
        selected = _selected_for_task(candidates, task.get("task_id"))
        benchmark = selected.get("benchmark_summary") if selected else {}
        lines.append(
            "| {task} | {verified} | {candidate} | {speedup} | {notes} |".format(
                task=task.get("task_id"),
                verified="yes" if (task.get("verification") or {}).get("passed") else "no",
                candidate=selected.get("candidate_id", "n/a") if selected else "n/a",
                speedup=_fmt(benchmark.get("speedup_vs_eager") if benchmark else None),
                notes=_task_notes(task),
            )
        )

    lines.extend(
        [
            "",
            "## Candidate-Level Outcome Table",
            "",
            "| Task | Candidate | Stage | Round | Parent speedup | Policy | Verified | Selected | Failure type | Speedup vs eager | Heuristic flags |",
            "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for record in candidates:
        source = read_artifact(record.get("candidate_path"), run_dir=run_dir)
        flags = static_performance_flags(record.get("task_id"), source)
        heuristic_counts.update(flags)
        classification = classifications[id(record)]
        benchmark = record.get("benchmark_summary") or {}
        lines.append(
            "| {task} | {candidate} | {stage} | {round} | {parent_speedup} | {policy} | {verified} | {selected} | {failure} | {speedup} | {flags} |".format(
                task=record.get("task_id"),
                candidate=record.get("candidate_id"),
                stage=record.get("generation_stage", "initial"),
                round=record.get("search_round") if record.get("search_round") is not None else "",
                parent_speedup=_fmt(record.get("parent_speedup_vs_eager")),
                policy="pass" if record.get("policy_passed") else "fail",
                verified="yes" if record.get("verification_passed") else "no",
                selected="yes" if record.get("selected_best") else "no",
                failure=classification.failure_type,
                speedup=_fmt(benchmark.get("speedup_vs_eager")),
                flags=", ".join(flags) if flags else "none",
            )
        )

    lines.extend(["", "## Best Candidate Per Task", ""])
    selected = [record for record in candidates if record.get("selected_best")]
    if selected:
        for record in selected:
            benchmark = record.get("benchmark_summary") or {}
            lines.append(
                f"- {record.get('task_id')} {record.get('candidate_id')}: "
                f"speedup_vs_eager={_fmt(benchmark.get('speedup_vs_eager'))}"
            )
    else:
        lines.append("No selected correct candidates were recorded.")

    slow = [record for record in candidates if record.get("verification_passed") and _speedup(record) < 1.0]
    lines.extend(["", "## Slow-But-Correct Candidate Analysis", ""])
    if slow:
        for record in slow:
            source = read_artifact(record.get("candidate_path"), run_dir=run_dir)
            flags = static_performance_flags(record.get("task_id"), source)
            lines.append(
                f"- {record.get('task_id')} {record.get('candidate_id')}: "
                f"speedup_vs_eager={_fmt(_speedup(record))}; "
                f"heuristics={', '.join(flags) if flags else 'none'}"
            )
    else:
        lines.append("No slow-but-correct candidates detected.")

    compile_failures = [
        record for record in candidates if classifications[id(record)].failure_type == "TRITON_COMPILE_ERROR"
    ]
    lines.extend(["", "## Triton Compile Error Analysis", ""])
    if compile_failures:
        for record in compile_failures:
            log_text = read_artifact(record.get("error_log_path"), run_dir=run_dir)
            lines.append(f"- {record.get('task_id')} {record.get('candidate_id')}: {_first_error_line(log_text)}")
    else:
        lines.append("No Triton compile errors detected.")

    lines.extend(["", "## Common Performance Problems Detected Statically", ""])
    if heuristic_counts:
        for flag, count in heuristic_counts.most_common():
            lines.append(f"- {flag}: {count}")
        lines.append("")
        lines.append("These are static heuristics, not profiler facts.")
    else:
        lines.append("No obvious static performance issues detected.")

    lines.extend(["", "## Prompt Weaknesses", ""])
    lines.extend(_prompt_weaknesses(candidates, classifications, heuristic_counts))

    lines.extend(["", "## Repair-Prompt Weaknesses", ""])
    lines.extend(_repair_prompt_weaknesses(candidates, classifications))

    lines.extend(["", "## Dataset Usefulness", ""])
    lines.extend(_dataset_usefulness(candidates, classifications))

    lines.extend(["", "## Concrete Recommendations For Next Prompt Version", ""])
    lines.extend(_recommendations(heuristic_counts, compile_failures, slow))
    lines.append("")
    return "\n".join(lines)


def static_performance_flags(task_id: Any, source: str) -> list[str]:
    """Return static heuristic flags for likely GPU performance problems."""

    flags: list[str] = []
    lowered = source.lower()
    block_sizes = [int(value) for value in re.findall(r"BLOCK_SIZE\s*=\s*(\d+)", source)]
    block_sizes += [int(value) for value in re.findall(r"BLOCK\s*=\s*(\d+)", source)]
    if block_sizes and min(block_sizes) < 128:
        flags.append("too small BLOCK_SIZE")
    if "tl.arange" not in source and "program_id" in source:
        flags.append("one element per program instead of block vectorization")
    if "except importerror" in lowered or "except module" in lowered:
        flags.append("Python fallback branch")
    if "reshape(-1)" not in source and ".flatten(" not in source and "view(-1)" not in source:
        flags.append("not using contiguous flattening")
    if str(task_id) == "bias_relu" and "% features" not in lowered and "tl.arange" in source:
        flags.append("poor or unclear bias indexing")
    if source.count(".shape") > 3 or source.count(".numel()") > 2:
        flags.append("repeated shape computation in wrapper")
    if "tl.constexpr" not in source:
        flags.append("not using constexpr for block size")
    if "@triton.jit" not in source:
        flags.append("not using triton.jit correctly")
    if _uses_unnecessary_torch_ops(source):
        flags.append("torch ops in forward path beyond allocation/wrapping")
    if "mask =" in source and "power-of-two" in lowered:
        flags.append("unnecessary masks for power-of-two shape")
    if "torch.empty_like" not in source and "torch.empty(" not in source:
        flags.append("unnecessary intermediate tensors or unclear output allocation")
    return flags


def _uses_unnecessary_torch_ops(source: str) -> bool:
    allowed = {"empty", "empty_like", "empty_strided", "as_strided", "reshape", "view", "contiguous"}
    for match in re.finditer(r"torch\.([A-Za-z_][A-Za-z0-9_]*)", source):
        if match.group(1) not in allowed:
            return True
    return False


def _selected_for_task(candidates: list[dict[str, Any]], task_id: Any) -> dict[str, Any]:
    return next(
        (record for record in candidates if record.get("task_id") == task_id and record.get("selected_best")),
        {},
    )


def _task_notes(task: dict[str, Any]) -> str:
    if (task.get("verification") or {}).get("passed"):
        return ""
    return task.get("failure_reason") or "not verified"


def _speedup(record: dict[str, Any]) -> float:
    value = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
    return float(value) if value is not None else 0.0


def _first_error_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and ("Error" in stripped or "Exception" in stripped or "failed" in stripped.lower()):
            return stripped[:240]
    return "See error log."


def _prompt_weaknesses(candidates: list[dict[str, Any]], classifications: dict[int, Any], flags: Counter[str]) -> list[str]:
    lines = []
    if flags["not using contiguous flattening"]:
        lines.append("- Prompts should explicitly ask for flattening simple elementwise tensors.")
    if flags["poor or unclear bias indexing"]:
        lines.append("- `bias_relu` needs stronger last-dimension indexing guidance.")
    if flags["Python fallback branch"]:
        lines.append("- Prompt should forbid `try/except ImportError` torch fallback branches when fallback is disabled.")
    if any(classifications[id(record)].failure_type == "TRITON_COMPILE_ERROR" for record in candidates):
        lines.append("- Prompt should remind the model about Triton JIT restrictions and constexpr arguments.")
    if not lines:
        lines.append("- No obvious prompt weakness detected by static heuristics.")
    return lines


def _repair_prompt_weaknesses(candidates: list[dict[str, Any]], classifications: dict[int, Any]) -> list[str]:
    has_slow = any(record.get("verification_passed") and _speedup(record) < 1.0 for record in candidates)
    has_compile = any(classification.failure_type == "TRITON_COMPILE_ERROR" for classification in classifications.values())
    lines = []
    if has_slow:
        lines.append("- Repair prompts should explicitly say correctness passed but the candidate is slower than eager.")
    if has_compile:
        lines.append("- Repair prompts should include Triton compile traceback guidance and JIT restrictions.")
    if not lines:
        lines.append("- No repair-specific weakness detected.")
    return lines


def _dataset_usefulness(candidates: list[dict[str, Any]], classifications: dict[int, Any]) -> list[str]:
    sft = sum(1 for record in candidates if record.get("policy_passed") and record.get("verification_passed"))
    repair = sum(1 for classification in classifications.values() if classification.suggested_dataset_split == "repair")
    optimization = sum(1 for classification in classifications.values() if classification.suggested_dataset_split == "optimization")
    rejected = sum(1 for record in candidates if not record.get("verification_passed"))
    return [
        f"- SFT usefulness: {sft} verified candidates, all require human review.",
        f"- Repair usefulness: {repair} failed candidates with repair-oriented labels.",
        f"- Optimization usefulness: {optimization} correct-but-not-fast candidates.",
        f"- Rejected/failure analysis: {rejected} candidates.",
    ]


def _recommendations(
    flags: Counter[str],
    compile_failures: list[dict[str, Any]],
    slow: list[dict[str, Any]],
) -> list[str]:
    lines = []
    if slow:
        lines.append("- Add task-specific Triton skeletons with block vectorization and flattening.")
        lines.append("- Continue generation after first correct candidate to collect faster alternatives.")
    if compile_failures:
        lines.append("- Add CUDA-aware compile-error repair guidance for Triton JIT restrictions.")
    if flags["poor or unclear bias indexing"]:
        lines.append("- For `bias_relu`, state `feature_idx = offsets % features` directly.")
    if not lines:
        lines.append("- Keep collecting GPU runs before changing prompts further.")
    return lines


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
    return "n/a" if value is None else f"{float(value):.3f}"
