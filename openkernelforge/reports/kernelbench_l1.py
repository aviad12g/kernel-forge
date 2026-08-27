"""Reports and checks for the KernelBench L1 pilot."""

from __future__ import annotations

import dataclasses
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from openkernelforge.agents.backends import create_backend
from openkernelforge.agents.code_extract import extract_python_code
from openkernelforge.config import RunConfig, load_config, redact_secrets
from openkernelforge.harness.benchmarker import BenchmarkResult, benchmark_task
from openkernelforge.harness.policy import CandidatePolicyResult, check_candidate_policy
from openkernelforge.harness.sandbox import (
    load_candidate_from_path,
    resolve_task_artifact_dir,
    unload_candidate,
    write_candidate_source,
)
from openkernelforge.harness.verifier import VerificationResult, verify_candidate
from openkernelforge.tasks.kernelbench_l1 import (
    KernelBenchL1Error,
    bind_kernelbench_candidate,
    estimate_task_memory,
    generate_inputs_for_memory_estimate,
    load_kernelbench_l1_tasks,
    make_candidate_provider,
)
from openkernelforge.utils.env_probe import format_environment_summary, probe_environment
from openkernelforge.utils.gpu import dtype_from_name


def run_kernelbench_l1_check(config_path: str | Path, kernelbench_dir: str | Path) -> Path:
    config = load_config(config_path)
    raw_config = _load_raw_yaml(config_path)
    kernelbench_cfg = raw_config.get("kernelbench", {}) if isinstance(raw_config, dict) else {}
    max_tasks = int(kernelbench_cfg.get("max_tasks", 5))
    selected_task_ids = list(kernelbench_cfg.get("task_ids") or [])
    stratify_by_family = bool(kernelbench_cfg.get("stratify_by_family", False))
    record_skipped_tasks = bool(kernelbench_cfg.get("record_skipped_tasks", True))
    max_numel_per_input = _optional_int(kernelbench_cfg.get("max_numel_per_input"))
    max_total_input_bytes = _optional_int(kernelbench_cfg.get("max_total_input_bytes"))
    max_memory_mb = _optional_int(kernelbench_cfg.get("skip_if_estimated_memory_gt_mb"))
    allow_cpu_memory_preflight_fallback = bool(
        kernelbench_cfg.get("allow_cpu_memory_preflight_fallback", False)
    )
    memory_filter_enabled = any(
        value is not None for value in (max_numel_per_input, max_total_input_bytes, max_memory_mb)
    )
    provider = make_candidate_provider(kernelbench_cfg)
    repair_index = _load_repair_index(kernelbench_cfg) if provider.mode == "gemini_repair" else {}
    if provider.mode == "gemini_repair" and not selected_task_ids:
        selected_task_ids = list(repair_index)
        max_tasks = min(max_tasks, len(selected_task_ids)) if selected_task_ids else max_tasks
    if provider.mode == "llm_later":
        provider.candidate_for_task("__probe__")

    run_dir = _new_run_dir(config.output_dir)
    for folder in ("prompts", "responses", "candidates", "logs"):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(redact_secrets(raw_config), sort_keys=False),
        encoding="utf-8",
    )
    env = probe_environment()
    (run_dir / "environment_probe.json").write_text(
        json.dumps(env.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        _check_execution_requirements(config, env)
        _validate_kernelbench_run_config(
            config,
            kernelbench_cfg,
            provider_mode=provider.mode,
            max_tasks=max_tasks,
            max_numel_per_input=max_numel_per_input,
            max_total_input_bytes=max_total_input_bytes,
            max_memory_mb=max_memory_mb,
        )
        tasks = load_kernelbench_l1_tasks(
            kernelbench_dir,
            task_ids=selected_task_ids or None,
            max_tasks=None if memory_filter_enabled and not selected_task_ids else max_tasks,
            stratify_by_family=stratify_by_family,
        )
        if provider.mode == "gemini_repair":
            _validate_repair_contracts(repair_index, tasks)
        backend = create_backend(config.agent) if provider.mode in {"gemini", "gemini_repair"} else None
    except Exception as exc:
        report_path = run_dir / "kernelbench_l1_check.md"
        failure_data = {
            "run_dir": str(run_dir),
            "config": str(config_path),
            "kernelbench_dir": str(kernelbench_dir),
            "status": "failed",
            "error": str(exc),
            "environment": env.to_dict(),
            "records": [],
        }
        (run_dir / "kernelbench_l1_check.json").write_text(
            json.dumps(failure_data, indent=2) + "\n"
        )
        write_kernelbench_l1_report(run_dir, data=failure_data)
        return report_path

    device = _select_device(config)
    selected_count = 0
    scanned_count = 0
    skipped_count_total = 0
    skipped_reason_counts: dict[str, int] = {}
    results_path = run_dir / "results.jsonl"
    results_file = results_path.open("w", encoding="utf-8")
    for task in tasks:
        if selected_count >= max_tasks:
            break
        scanned_count += 1
        record: dict[str, Any] = {
            "task_id": task.task_id,
            "task_name": task.name,
            "op_family": task.metadata.get("op_family"),
            "shape": list(task.benchmark_shapes[0]),
            "source_path": task.metadata.get("source_path"),
            "source_relative_path": task.metadata.get("source_relative_path"),
            "module_name": task.metadata.get("module_name"),
            "candidate_contract": task.metadata.get("candidate_contract"),
            "reference_has_model_state": task.metadata.get("reference_has_model_state"),
            "candidate_provider": provider.mode,
            "candidate_path": None,
            "reference_ok": False,
            "benchmark_summary": None,
            "candidate_record": None,
            "input_memory_estimate": None,
            "skipped": False,
            "skip_reason": None,
            "error": None,
        }
        try:
            provider_path = provider.candidate_for_task(task.task_id)
            record["candidate_path"] = str(provider_path) if provider_path else None
            if memory_filter_enabled:
                try:
                    estimate_inputs = generate_inputs_for_memory_estimate(
                        task,
                        seed=0,
                        dtype=torch.float32,
                        allow_cpu_fallback=allow_cpu_memory_preflight_fallback,
                    )
                except KernelBenchL1Error as exc:
                    record["skipped"] = True
                    record["skip_reason"] = "MEMORY_ESTIMATE_UNAVAILABLE"
                    record["error"] = str(exc)
                    skipped_count_total += 1
                    skipped_reason_counts["MEMORY_ESTIMATE_UNAVAILABLE"] = (
                        skipped_reason_counts.get("MEMORY_ESTIMATE_UNAVAILABLE", 0) + 1
                    )
                    if record_skipped_tasks:
                        records.append(record)
                    continue
                cache_overhead = (
                    int(config.benchmark.cache_flush.size_mb) * 1024 * 1024
                    if config.benchmark.cache_flush.enabled
                    else 0
                )
                estimate = estimate_task_memory(
                    task,
                    estimate_inputs,
                    known_overhead_bytes=cache_overhead,
                )
                _record_materialized_input_shapes(task, estimate, record)
                record["input_memory_estimate"] = estimate
                skip_reason = _memory_skip_reason(
                    estimate,
                    max_numel_per_input=max_numel_per_input,
                    max_total_input_bytes=max_total_input_bytes,
                    max_memory_mb=max_memory_mb,
                )
                if skip_reason:
                    record["skipped"] = True
                    record["skip_reason"] = skip_reason
                    skipped_count_total += 1
                    skipped_reason_counts[skip_reason] = skipped_reason_counts.get(skip_reason, 0) + 1
                    if record_skipped_tasks:
                        records.append(record)
                    continue
            selected_count += 1
            inputs = task.generate_inputs(
                0,
                task.benchmark_shapes[0],
                torch.float32,
                device,
            )
            if not record.get("input_shapes"):
                _record_materialized_input_shapes(task, estimate_task_memory(task, inputs), record)
            with torch.no_grad():
                output = task.reference_fn(*inputs)
            record["reference_ok"] = True
            record["output_summary"] = _output_summary(output)
            if config.benchmark.enabled:
                benchmark = benchmark_task(
                    task,
                    task.reference_fn,
                    candidate_name="reference_baseline",
                    shape=task.benchmark_shapes[0],
                    dtype=config.benchmark.dtype,
                    device=device,
                    warmup=config.benchmark.warmup,
                    repeats=config.benchmark.repeats,
                    timing_mode=config.benchmark.timing_mode,
                    independent_sessions=config.benchmark.independent_sessions,
                    cache_flush_config=config.benchmark.cache_flush,
                    bootstrap_ci_config=config.benchmark.bootstrap_ci,
                    separate_compile_time=config.benchmark.separate_compile_time,
                    stable_session_threshold=config.benchmark.stable_session_threshold,
                    enable_torch_compile=config.benchmark.enable_torch_compile,
                    torch_compile_mode=config.benchmark.torch_compile_mode,
                )
                benchmark_dict = _benchmark_to_dict(benchmark)
                record["benchmark_summary"] = _benchmark_summary(benchmark_dict)
                if benchmark.benchmark_error:
                    failures.append(f"{task.task_id}: baseline benchmark failed")
                if config.benchmark.enable_torch_compile and benchmark.compile_error:
                    failures.append(f"{task.task_id}: torch.compile baseline failed")
            if provider.mode in {"gemini", "gemini_repair"} and backend is not None:
                repair_info = repair_index.get(task.task_id) if provider.mode == "gemini_repair" else None
                if provider.mode == "gemini_repair" and repair_info is None:
                    records.append(record)
                    continue
                candidate_record = _generate_and_evaluate_candidate(
                    task=task,
                    run_dir=run_dir,
                    config=config,
                    backend=backend,
                    device=device,
                    candidate_index=0,
                    provider_mode=provider.mode,
                    repair_info=repair_info,
                )
                record["candidate_record"] = candidate_record
                candidate_records.append(candidate_record)
                results_file.write(json.dumps(candidate_record) + "\n")
                results_file.flush()
            elif provider.mode == "existing_file":
                if provider_path is None:
                    raise KernelBenchL1Error(
                        f"No existing candidate file found for task {task.task_id} under {provider.root}"
                    )
                candidate_record = _evaluate_existing_candidate(
                    task=task,
                    run_dir=run_dir,
                    config=config,
                    device=device,
                    candidate_index=0,
                    source_path=provider_path,
                )
                record["candidate_record"] = candidate_record
                candidate_records.append(candidate_record)
                results_file.write(json.dumps(candidate_record) + "\n")
                results_file.flush()
        except Exception as exc:
            record["error"] = str(exc)
            failures.append(f"{task.task_id}: {exc}")
        records.append(record)
    results_file.close()

    data: dict[str, Any] = {
        "run_dir": str(run_dir),
        "config": str(config_path),
        "kernelbench_dir": str(kernelbench_dir),
        "status": "completed" if not failures else "completed_with_failures",
        "environment": env.to_dict(),
        "tasks_loaded": len(tasks),
        "tasks_scanned": scanned_count,
        "tasks_attempted": scanned_count,
        "tasks_selected": selected_count,
        "tasks_skipped": skipped_count_total,
        "skipped_reasons": skipped_reason_counts,
        "records": records,
        "candidate_records": candidate_records,
        "failures": failures,
        "timing": {
            "timing_mode": config.benchmark.timing_mode,
            "warmup": config.benchmark.warmup,
            "repeat": config.benchmark.repeats,
            "independent_sessions": config.benchmark.independent_sessions,
            "cache_flush_enabled": config.benchmark.cache_flush.enabled,
            "bootstrap_ci_enabled": config.benchmark.bootstrap_ci.enabled,
            "torch_compile_enabled": config.benchmark.enable_torch_compile,
            "torch_compile_mode": config.benchmark.torch_compile_mode,
        },
        "kernelbench_selection": {
            "max_tasks": max_tasks,
            "stratify_by_family": stratify_by_family,
            "record_skipped_tasks": record_skipped_tasks,
            "max_numel_per_input": max_numel_per_input,
            "max_total_input_bytes": max_total_input_bytes,
            "skip_if_estimated_memory_gt_mb": max_memory_mb,
            "allow_cpu_memory_preflight_fallback": allow_cpu_memory_preflight_fallback,
            "selection_rule": (
                "first_feasible_tasks_in_family_round_robin_order"
                if stratify_by_family
                else "first_feasible_tasks_in_discovery_order"
            ),
            "pool_scan_complete": scanned_count >= len(tasks),
            "memory_skip_count_scope": (
                "loaded_pool"
                if scanned_count >= len(tasks)
                else "scanned_prefix_before_selection_cap"
            ),
        },
    }
    (run_dir / "kernelbench_l1_check.json").write_text(json.dumps(data, indent=2) + "\n")
    return write_kernelbench_l1_report(run_dir, data=data)


def write_kernelbench_l1_report(
    run_dir: str | Path,
    *,
    data: dict[str, Any] | None = None,
) -> Path:
    run_path = Path(run_dir)
    if data is None:
        json_path = run_path / "kernelbench_l1_check.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))
    report_path = run_path / "kernelbench_l1_pilot_report.md"
    check_path = run_path / "kernelbench_l1_check.md"
    records = data.get("records") or []
    candidate_records = data.get("candidate_records") or []
    selected_count = data.get("tasks_selected")
    if selected_count is None:
        selected_count = sum(1 for record in records if not record.get("skipped"))
    skipped_count = data.get("tasks_skipped")
    if skipped_count is None:
        skipped_count = sum(1 for record in records if record.get("skipped"))
    lines = [
        "# KernelBench L1 Pilot Report",
        "",
        "This report records KernelBench L1 task loading, baseline timing, and optional candidate generation. It contains no training, no RL, and no SOTA claim.",
        "",
        "## Summary",
        "",
        f"- Status: `{data.get('status', 'unknown')}`",
        f"- Run dir: `{data.get('run_dir', run_path)}`",
        f"- KernelBench dir: `{data.get('kernelbench_dir', 'n/a')}`",
        f"- Tasks loaded: {data.get('tasks_loaded', len(records))}",
        f"- Tasks scanned before reaching the selection cap: {data.get('tasks_scanned', data.get('tasks_attempted', len(records)))}",
        f"- Tasks selected for timing: {selected_count}",
        f"- Tasks skipped before timing: {skipped_count}",
        f"- Candidate results present: {len(candidate_records)}",
    ]
    timing = data.get("timing") or {}
    if timing:
        timed_denominator = max(int(selected_count or 0), 0)
        eager_timed = sum(
            1
            for record in records
            if ((record.get("benchmark_summary") or {}).get("eager_median_ms") is not None)
        )
        compile_timed = sum(
            1
            for record in records
            if ((record.get("benchmark_summary") or {}).get("torch_compile_median_ms") is not None)
        )
        benchmark_failures = sum(
            1
            for record in records
            if not record.get("skipped")
            and ((record.get("benchmark_summary") or {}).get("benchmark_error") or record.get("error"))
        )
        compile_failures = sum(
            1
            for record in records
            if (record.get("benchmark_summary") or {}).get("compile_error")
        )
        lines.extend(
            [
                f"- Timing mode: `{timing.get('timing_mode')}`",
                f"- Cache flush enabled: {timing.get('cache_flush_enabled')}",
                f"- Independent sessions: {timing.get('independent_sessions')}",
                f"- Repeat: {timing.get('repeat')}",
                f"- Bootstrap CI enabled: {timing.get('bootstrap_ci_enabled')}",
                f"- torch.compile enabled: {timing.get('torch_compile_enabled')}",
                f"- Eager timed successfully: {eager_timed}/{timed_denominator or len(records)}",
                f"- torch.compile timed successfully: {compile_timed}/{timed_denominator or len(records)}",
                f"- Benchmark failures: {benchmark_failures}",
                f"- torch.compile failures: {compile_failures}",
            ]
        )
    skipped_reasons = data.get("skipped_reasons") or _count_skipped_reasons(records)
    selection = data.get("kernelbench_selection") or {}
    if selection:
        lines.extend(
            [
                f"- Selection rule: `{selection.get('selection_rule', 'not recorded')}`",
                f"- Loaded-pool scan complete: {selection.get('pool_scan_complete', 'not recorded')}",
                f"- Memory-skip count scope: `{selection.get('memory_skip_count_scope', 'not recorded')}`",
            ]
        )
    if skipped_reasons:
        lines.extend(["", "## Skipped Tasks", ""])
        for reason, count in sorted(skipped_reasons.items()):
            lines.append(f"- `{reason}`: {count}")
    if data.get("error"):
        lines.extend(["", "## Error", "", str(data["error"])])

    lines.extend(["", "## Environment", "", "```text"])
    env = data.get("environment")
    if env:
        lines.append(format_environment_summary(env))
    else:
        lines.append("not recorded")
    lines.extend(["```", "", "## Tasks", ""])
    lines.append("| Task | Op family | Shape | Eager median ms | torch.compile median ms | Status | Failure / skip reason |")
    lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
    for record in records:
        summary = record.get("benchmark_summary") or {}
        status = _record_status(record, summary)
        failure = (
            record.get("skip_reason")
            or record.get("error")
            or summary.get("benchmark_error")
            or summary.get("compile_error")
            or ""
        )
        lines.append(
            "| `{task}` | {family} | `{shape}` | {eager} | {compile} | {status} | {failure} |".format(
                task=record.get("task_id"),
                family=record.get("op_family") or "",
                shape=record.get("shape"),
                eager=_fmt(summary.get("eager_median_ms")),
                compile=_fmt(summary.get("torch_compile_median_ms")),
                status=status,
                failure=_short_failure(failure),
            )
        )
    lines.extend(["", "## Candidate Results", ""])
    if candidate_records:
        lines.extend(_candidate_report_lines(candidate_records))
    else:
        lines.extend(
            [
                "Candidate generation is intentionally optional in this sprint. `candidate_provider=none` validates task loading, eager references, and optional `torch.compile` baselines only.",
                "",
                "- Single-run wins: none recorded by baseline-only check.",
                "- Repeat-stable wins: none recorded by baseline-only check.",
                "- Single-run-only wins: none recorded by baseline-only check.",
                "- Unstable fraction: not applicable without candidate results.",
                "- Failure taxonomy: reference/load/benchmark failures are reported above.",
            ]
        )
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            "Ready for candidate evaluation only when every selected task is `ready`, requested compiler baselines succeeded, and timing summaries are present.",
        ]
    )
    text = "\n".join(lines) + "\n"
    report_path.write_text(text, encoding="utf-8")
    check_path.write_text(text, encoding="utf-8")
    return check_path


def _load_raw_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _load_repair_index(kernelbench_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path_value = (
        kernelbench_cfg.get("repair_subset_path")
        or kernelbench_cfg.get("repair_taxonomy_path")
        or kernelbench_cfg.get("failure_taxonomy_path")
    )
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        raise KernelBenchL1Error(f"candidate_provider=gemini_repair requires existing repair taxonomy: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_items = data.get("selected_for_repair") or data.get("failures") or []
    requested = set(str(item) for item in kernelbench_cfg.get("repair_task_ids") or [])
    index: dict[str, dict[str, Any]] = {}
    current_policy_version = CandidatePolicyResult(passed=False).policy_version
    invalid_provenance: list[str] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or "")
        if not task_id:
            continue
        if requested and task_id not in requested:
            continue
        if (
            item.get("parent_policy_version") != current_policy_version
            or not item.get("parent_candidate_contract")
        ):
            invalid_provenance.append(task_id)
            continue
        index[task_id] = item
    if invalid_provenance:
        raise KernelBenchL1Error(
            "Repair taxonomy contains historical or unversioned parent candidates: "
            + ", ".join(sorted(invalid_provenance)[:8])
            + f". Regenerate failures with policy {current_policy_version} and the corrected candidate contract."
        )
    if requested:
        missing = sorted(requested.difference(index))
        if missing:
            raise KernelBenchL1Error(
                "Repair taxonomy missing requested task ids: " + ", ".join(missing[:8])
            )
    return index


def _validate_repair_contracts(
    repair_index: dict[str, dict[str, Any]],
    tasks: list[Any],
) -> None:
    tasks_by_id = {str(task.task_id): task for task in tasks}
    mismatches: list[str] = []
    for task_id, item in repair_index.items():
        task = tasks_by_id.get(task_id)
        if task is None:
            mismatches.append(f"{task_id}:task_not_loaded")
            continue
        expected = str(task.metadata.get("candidate_contract") or "forward")
        actual = str(item.get("parent_candidate_contract") or "")
        if actual != expected:
            mismatches.append(f"{task_id}:{actual or 'missing'}!={expected}")
    if mismatches:
        raise KernelBenchL1Error(
            "Repair parent candidate contract does not match the loaded task contract: "
            + ", ".join(mismatches[:8])
        )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _memory_skip_reason(
    estimate: dict[str, Any],
    *,
    max_numel_per_input: int | None,
    max_total_input_bytes: int | None,
    max_memory_mb: int | None,
) -> str | None:
    if max_numel_per_input is not None and int(estimate.get("max_tensor_numel") or 0) > max_numel_per_input:
        return "ESTIMATED_INPUT_NUMEL_TOO_LARGE"
    if max_total_input_bytes is not None and int(estimate.get("total_bytes") or 0) > max_total_input_bytes:
        return "ESTIMATED_INPUT_BYTES_TOO_LARGE"
    if max_memory_mb is not None:
        limit = int(max_memory_mb) * 1024 * 1024
        estimated_bytes = int(
            estimate.get("estimated_known_peak_bytes")
            or estimate.get("minimum_resident_bytes")
            or estimate.get("total_bytes")
            or 0
        )
        if estimated_bytes > limit:
            return "ESTIMATED_MEMORY_TOO_LARGE"
    return None


def _count_skipped_reasons(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if not record.get("skipped"):
            continue
        reason = str(record.get("skip_reason") or "SKIPPED")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _record_materialized_input_shapes(
    task: Any,
    estimate: dict[str, Any],
    record: dict[str, Any],
) -> None:
    shapes = [list(item.get("shape") or []) for item in estimate.get("tensors") or []]
    record["input_shapes"] = shapes
    task.metadata["materialized_input_shapes"] = shapes
    if task.benchmark_shapes and not task.benchmark_shapes[0] and shapes:
        inferred = tuple(int(dim) for dim in shapes[0])
        task.benchmark_shapes[0] = inferred
        task.metadata["shape_metadata"] = {
            "shape": list(inferred),
            "rank": len(inferred),
            "numel": int(estimate.get("tensors", [{}])[0].get("numel") or 0),
            "inference_status": "materialized_input",
        }
        record["shape"] = list(inferred)


def _record_status(record: dict[str, Any], summary: dict[str, Any]) -> str:
    if record.get("skipped"):
        return "skipped"
    if (
        record.get("reference_ok")
        and not record.get("error")
        and summary.get("eager_median_ms") is not None
        and not summary.get("benchmark_error")
    ):
        if summary.get("compile_error"):
            return "partial"
        return "ready"
    return "failed"


def _generate_and_evaluate_candidate(
    *,
    task: Any,
    run_dir: Path,
    config: RunConfig,
    backend: Any,
    device: str,
    candidate_index: int,
    provider_mode: str = "gemini",
    repair_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    is_repair = provider_mode == "gemini_repair"
    prompt = (
        _build_kernelbench_repair_prompt(task, repair_info or {})
        if is_repair
        else build_kernelbench_prompt(task)
    )
    prompt_path = _write_task_artifact(
        run_dir,
        "prompts",
        task.task_id,
        candidate_index,
        "_prompt.txt",
        prompt,
    )
    try:
        raw_response = backend.generate(
            prompt,
            system=(
                "You generate concise Python candidate kernels for local verification. "
                "Return only a Python code block."
            ),
            **_generation_kwargs(config),
        )
    except Exception:
        return _generation_failure_record(
            task=task,
            run_dir=run_dir,
            config=config,
            candidate_index=candidate_index,
            provider_mode=provider_mode,
            prompt_path=prompt_path,
            repair_info=repair_info,
            error_text=traceback.format_exc(),
        )
    response_path = _write_task_artifact(
        run_dir,
        "responses",
        task.task_id,
        candidate_index,
        "_response.txt",
        raw_response,
    )
    extraction = extract_python_code(raw_response)
    candidate_source = extraction.code or "# Code extraction failed. See response artifact.\n"
    candidate_path = write_candidate_source(run_dir, task.task_id, candidate_index, candidate_source)
    return _evaluate_candidate_artifact(
        task=task,
        run_dir=run_dir,
        config=config,
        device=device,
        candidate_index=candidate_index,
        candidate_source=candidate_source,
        candidate_path=candidate_path,
        source_type="gemini_repair" if is_repair else "gemini",
        generation_stage=(
            "kernelbench_l1_gemini_repair1" if is_repair else "kernelbench_l1_gemini_pilot"
        ),
        backend=config.agent.backend,
        model=config.agent.model,
        prompt_path=prompt_path,
        response_path=response_path,
        extraction=extraction,
        extraction_error=extraction.error,
        repair_info=repair_info,
    )


def _generation_failure_record(
    *,
    task: Any,
    run_dir: Path,
    config: RunConfig,
    candidate_index: int,
    provider_mode: str,
    prompt_path: Path,
    repair_info: dict[str, Any] | None,
    error_text: str,
) -> dict[str, Any]:
    source_type = "gemini_repair" if provider_mode == "gemini_repair" else "gemini"
    policy = CandidatePolicyResult(passed=False, rejection_reason="backend_generation_failed")
    verification = VerificationResult(
        task_id=task.task_id,
        candidate_name=f"kernelbench_{source_type}_{task.task_id}_{candidate_index:03d}",
        passed=False,
        error="backend_generation_failed",
    )
    error_log_path = _write_error_log(
        run_dir,
        task.task_id,
        candidate_index,
        ["Backend generation failed:\n" + error_text],
    )
    return {
        "record_type": "kernelbench_candidate",
        "task_id": task.task_id,
        "task_name": task.name,
        "task_family": "kernelbench_l1",
        "op_family": task.metadata.get("op_family"),
        "source_type": source_type,
        "generation_stage": (
            "kernelbench_l1_gemini_repair1"
            if provider_mode == "gemini_repair"
            else "kernelbench_l1_gemini_pilot"
        ),
        "backend": config.agent.backend,
        "model": config.agent.model,
        "candidate_index": candidate_index,
        "source_candidate_path": None,
        "parent_candidate_path": (repair_info or {}).get("candidate_path"),
        "parent_run_dir": (repair_info or {}).get("parent_run_dir"),
        "failure_category": (repair_info or {}).get("failure_category"),
        "repairability": (repair_info or {}).get("repairability"),
        "repair_instruction": (repair_info or {}).get("suggested_repair_instruction"),
        "prompt_path": str(prompt_path),
        "response_path": None,
        "candidate_path": None,
        "kernelbench_source_path": task.metadata.get("source_path"),
        "kernelbench_module_name": task.metadata.get("module_name"),
        "candidate_contract": task.metadata.get("candidate_contract"),
        "reference_has_model_state": task.metadata.get("reference_has_model_state"),
        "policy_version": policy.policy_version,
        "extraction": None,
        "policy": _to_jsonable(policy),
        "policy_passed": False,
        "policy_rejection_reason": policy.rejection_reason,
        "policy_warnings": [],
        "verification": _to_jsonable(verification),
        "verification_passed": False,
        "benchmark": None,
        "benchmark_summary": _candidate_benchmark_summary(None),
        "benchmarked": False,
        "candidate_label": "GENERATION_FAILED",
        "failure_reason": "backend_generation_failed",
        "error_log_path": error_log_path,
    }


def _evaluate_existing_candidate(
    *,
    task: Any,
    run_dir: Path,
    config: RunConfig,
    device: str,
    candidate_index: int,
    source_path: Path,
) -> dict[str, Any]:
    candidate_source = source_path.read_text(encoding="utf-8", errors="strict")
    candidate_path = write_candidate_source(
        run_dir,
        task.task_id,
        candidate_index,
        candidate_source,
    )
    return _evaluate_candidate_artifact(
        task=task,
        run_dir=run_dir,
        config=config,
        device=device,
        candidate_index=candidate_index,
        candidate_source=candidate_source,
        candidate_path=candidate_path,
        source_type="existing_file",
        generation_stage="kernelbench_l1_existing_file",
        backend="existing_file",
        model=None,
        source_candidate_path=source_path,
    )


def _evaluate_candidate_artifact(
    *,
    task: Any,
    run_dir: Path,
    config: RunConfig,
    device: str,
    candidate_index: int,
    candidate_source: str,
    candidate_path: Path,
    source_type: str,
    generation_stage: str,
    backend: str,
    model: str | None,
    prompt_path: Path | None = None,
    response_path: Path | None = None,
    extraction: Any | None = None,
    extraction_error: str | None = None,
    repair_info: dict[str, Any] | None = None,
    source_candidate_path: Path | None = None,
) -> dict[str, Any]:
    candidate_name = f"kernelbench_{source_type}_{task.task_id}_{candidate_index:03d}"
    error_chunks: list[str] = []
    policy = CandidatePolicyResult(passed=False, rejection_reason="not_evaluated")
    verification = VerificationResult(
        task_id=task.task_id,
        candidate_name=candidate_name,
        passed=False,
        error=None,
    )
    benchmark_dict: dict[str, Any] | None = None
    failure_reason: str | None = None
    loaded = None

    if extraction_error:
        error_chunks.append("Code extraction error:\n" + extraction_error)
        policy = CandidatePolicyResult(passed=False, rejection_reason="code_extraction_failed")
        verification.error = extraction_error
        failure_reason = "code_extraction_failed"
    else:
        policy = check_candidate_policy(
            candidate_source,
            allow_torch_fallback=config.agent.allow_torch_fallback,
            require_triton=not config.agent.allow_torch_fallback,
        )
        if not policy.passed:
            failure_reason = "policy_rejected"
            error_chunks.append("Policy rejected candidate:\n" + str(policy.rejection_reason))
        else:
            try:
                loaded = load_candidate_from_path(candidate_path, require_forward=False)
                candidate_callable = bind_kernelbench_candidate(
                    task,
                    loaded.module,
                    dtype=dtype_from_name(config.verification.dtype),
                    device=device,
                )
                verification = verify_candidate(
                    task,
                    candidate_callable,
                    candidate_name=candidate_name,
                    seeds=config.verification.seeds,
                    shapes=task.benchmark_shapes[: config.verification.max_shapes_per_task],
                    dtype=config.verification.dtype,
                    device=device,
                )
                if not verification.passed:
                    failure_reason = "verification_failed"
                    verification_text = verification.error or _verification_failure_summary(verification)
                    error_chunks.append("Verification failed:\n" + verification_text)
                elif config.benchmark.enabled:
                    benchmark = benchmark_task(
                        task,
                        candidate_callable,
                        candidate_name=candidate_name,
                        shape=task.benchmark_shapes[0],
                        dtype=config.benchmark.dtype,
                        device=device,
                        warmup=config.benchmark.warmup,
                        repeats=config.benchmark.repeats,
                        timing_mode=config.benchmark.timing_mode,
                        independent_sessions=config.benchmark.independent_sessions,
                        cache_flush_config=config.benchmark.cache_flush,
                        bootstrap_ci_config=config.benchmark.bootstrap_ci,
                        separate_compile_time=config.benchmark.separate_compile_time,
                        stable_session_threshold=config.benchmark.stable_session_threshold,
                        enable_torch_compile=config.benchmark.enable_torch_compile,
                        torch_compile_mode=config.benchmark.torch_compile_mode,
                    )
                    benchmark_dict = _benchmark_to_dict(benchmark)
                    if benchmark.benchmark_error:
                        failure_reason = "benchmark_failed"
                        error_chunks.append("Benchmark failed:\n" + benchmark.benchmark_error)
                    if config.benchmark.enable_torch_compile and benchmark.compile_error:
                        error_chunks.append("torch.compile failed:\n" + benchmark.compile_error)
            except Exception:
                failure_reason = "candidate_exception"
                error_chunks.append(traceback.format_exc())
            finally:
                if loaded is not None:
                    unload_candidate(loaded)

    benchmark_summary = _candidate_benchmark_summary(benchmark_dict)
    candidate_label = _candidate_label(
        policy_passed=policy.passed,
        verification_passed=verification.passed,
        benchmark_summary=benchmark_summary,
        failure_reason=failure_reason,
    )
    if failure_reason is None:
        failure_reason = _candidate_failure_reason(policy, verification, benchmark_summary)
    error_log_path = _write_error_log(run_dir, task.task_id, candidate_index, error_chunks)
    return {
        "record_type": "kernelbench_candidate",
        "task_id": task.task_id,
        "task_name": task.name,
        "task_family": "kernelbench_l1",
        "op_family": task.metadata.get("op_family"),
        "source_type": source_type,
        "generation_stage": generation_stage,
        "backend": backend,
        "model": model,
        "candidate_index": candidate_index,
        "source_candidate_path": str(source_candidate_path) if source_candidate_path else None,
        "parent_candidate_path": (repair_info or {}).get("candidate_path"),
        "parent_run_dir": (repair_info or {}).get("parent_run_dir"),
        "failure_category": (repair_info or {}).get("failure_category"),
        "repairability": (repair_info or {}).get("repairability"),
        "repair_instruction": (repair_info or {}).get("suggested_repair_instruction"),
        "prompt_path": str(prompt_path) if prompt_path else None,
        "response_path": str(response_path) if response_path else None,
        "candidate_path": str(candidate_path),
        "kernelbench_source_path": task.metadata.get("source_path"),
        "kernelbench_module_name": task.metadata.get("module_name"),
        "candidate_contract": task.metadata.get("candidate_contract"),
        "reference_has_model_state": task.metadata.get("reference_has_model_state"),
        "policy_version": policy.policy_version,
        "extraction": _to_jsonable(extraction),
        "policy": _to_jsonable(policy),
        "policy_passed": policy.passed,
        "policy_rejection_reason": policy.rejection_reason,
        "policy_warnings": list(policy.warnings),
        "verification": _to_jsonable(verification),
        "verification_passed": verification.passed,
        "benchmark": benchmark_dict,
        "benchmark_summary": benchmark_summary,
        "benchmarked": benchmark_summary.get("candidate_median_ms") is not None,
        "candidate_label": candidate_label,
        "failure_reason": failure_reason,
        "error_log_path": error_log_path,
    }


def build_kernelbench_prompt(task: Any) -> str:
    source_path = Path(str(task.metadata.get("source_path") or ""))
    source_text = ""
    if source_path.exists():
        source_text = source_path.read_text(encoding="utf-8", errors="replace")
    source_text = _truncate_middle(source_text, 16000)
    shape = task.benchmark_shapes[0]
    candidate_contract = str(task.metadata.get("candidate_contract") or "forward")
    runtime_contract = (
        "- ModelNew.forward receives the positional values returned by get_inputs().\n"
        "- ModelNew receives the constructor arguments returned by get_init_inputs().\n"
        if candidate_contract == "model_new"
        else "- forward receives the positional values returned by get_inputs().\n"
    )
    return (
        "Write one Python candidate for a KernelBench L1 task.\n"
        "Return only a fenced Python code block.\n"
        f"{_kernelbench_candidate_contract_text(candidate_contract)}\n\n"
        "Contract:\n"
        f"{runtime_contract}"
        "- It must match Model.forward for both the inputs and initialized model state.\n"
        "- Use Triton kernels when appropriate.\n"
        "- Do not call the PyTorch reference, Model class, get_inputs, or any KernelBench/OpenKernelForge task module during execution.\n"
        "- Do not fake outputs, cache expected tensors, read files, or use hidden state.\n"
        "- Torch fallback is disabled: avoid direct torch operations such as torch.matmul, torch.relu, torch.sum, torch.nn.functional.* as the main computation.\n"
        "- Torch may be used for allocation and wrappers, for example torch.empty, torch.empty_like, tensor shape/device/dtype inspection, and launching Triton kernels.\n"
        "- Include all imports required by the candidate.\n\n"
        f"Task id: {task.task_id}\n"
        f"Task name: {task.name}\n"
        f"Op family: {task.metadata.get('op_family')}\n"
        f"Candidate contract: {candidate_contract}\n"
        f"Reference has model state: {task.metadata.get('reference_has_model_state')}\n"
        f"Benchmark shape metadata: {task.metadata.get('shape_metadata')}\n"
        f"Benchmark shape: {shape}\n"
        f"Tolerance: rtol={task.tolerance.rtol}, atol={task.tolerance.atol}\n"
        f"KernelBench source path: {task.metadata.get('source_path')}\n\n"
        "KernelBench task source:\n"
        "```python\n"
        f"{source_text}\n"
        "```\n"
    )


def _build_kernelbench_repair_prompt(task: Any, repair_info: dict[str, Any]) -> str:
    source_path = Path(str(task.metadata.get("source_path") or ""))
    source_text = source_path.read_text(encoding="utf-8", errors="replace") if source_path.exists() else ""
    source_text = _truncate_middle(source_text, 14000)
    candidate_path = Path(str(repair_info.get("candidate_path") or ""))
    candidate_text = candidate_path.read_text(encoding="utf-8", errors="replace") if candidate_path.exists() else ""
    candidate_text = _truncate_middle(candidate_text, 10000)
    verification_error = _truncate_middle(str(repair_info.get("verification_error") or ""), 5000)
    traceback_summary = _truncate_middle(str(repair_info.get("traceback_summary") or ""), 4000)
    shape = task.benchmark_shapes[0]
    candidate_contract = str(task.metadata.get("candidate_contract") or "forward")
    runtime_contract = (
        "- Preserve ModelNew's constructor contract and forward input signature.\n"
        if candidate_contract == "model_new"
        else "- Preserve the module-level forward input signature.\n"
    )
    return (
        "Repair one failed Python candidate for a KernelBench L1 task.\n"
        "Return only a fenced Python code block with the corrected candidate.\n"
        f"{_kernelbench_candidate_contract_text(candidate_contract)}\n\n"
        "Repair objective:\n"
        "- Fix correctness first. Performance is secondary.\n"
        f"{runtime_contract}"
        "- Preserve the output contract.\n"
        "- Do not call the PyTorch reference, Model class, get_inputs, or any KernelBench/OpenKernelForge task module inside forward.\n"
        "- Do not use torch operations as the main computation. Torch is allowed only for allocation/wrapping, tensor shape/device/dtype inspection, and launching Triton kernels.\n"
        "- Include all imports required by the candidate.\n"
        "- If the failed candidate misunderstood the operation, replace it with a simpler correct Triton implementation rather than patching around the symptom.\n\n"
        f"Task id: {task.task_id}\n"
        f"Task name: {task.name}\n"
        f"Op family: {task.metadata.get('op_family')}\n"
        f"Candidate contract: {candidate_contract}\n"
        f"Reference has model state: {task.metadata.get('reference_has_model_state')}\n"
        f"Benchmark shape metadata: {task.metadata.get('shape_metadata')}\n"
        f"Benchmark shape: {shape}\n"
        f"Tolerance: rtol={task.tolerance.rtol}, atol={task.tolerance.atol}\n"
        f"KernelBench source path: {task.metadata.get('source_path')}\n"
        f"Failure category: {repair_info.get('failure_category')}\n"
        f"Suggested repair instruction: {repair_info.get('suggested_repair_instruction')}\n\n"
        "Verification error summary:\n"
        "```text\n"
        f"{verification_error}\n"
        "```\n\n"
        "Traceback / failure details:\n"
        "```text\n"
        f"{traceback_summary}\n"
        "```\n\n"
        "Original failed candidate:\n"
        "```python\n"
        f"{candidate_text}\n"
        "```\n\n"
        "KernelBench task source:\n"
        "```python\n"
        f"{source_text}\n"
        "```\n"
    )


def _kernelbench_candidate_contract_text(candidate_contract: str) -> str:
    if candidate_contract == "model_new":
        return (
            "The candidate file must define class ModelNew(torch.nn.Module) with the same "
            "constructor signature as Model and a forward method. Official KernelBench Model "
            "tasks do not accept a module-level forward function, including when Model has an "
            "empty state_dict."
        )
    if candidate_contract == "forward":
        return "The candidate file must define one module-level function: def forward(*args)."
    raise KernelBenchL1Error(f"Unsupported candidate contract: {candidate_contract}")


def _generation_kwargs(config: RunConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.agent.temperature is not None:
        kwargs["temperature"] = config.agent.temperature
    if config.agent.top_p is not None:
        kwargs["top_p"] = config.agent.top_p
    if config.agent.max_tokens is not None:
        kwargs["max_tokens"] = config.agent.max_tokens
    return kwargs


def _write_task_artifact(
    run_dir: Path,
    folder: str,
    task_id: str,
    candidate_index: int,
    suffix: str,
    text: str,
) -> Path:
    if candidate_index < 0:
        raise ValueError("candidate_index must be non-negative")
    task_dir = resolve_task_artifact_dir(run_dir, folder, task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"candidate_{candidate_index:03d}{suffix}"
    path.write_text(text, encoding="utf-8")
    return path


def _write_error_log(run_dir: Path, task_id: str, candidate_index: int, chunks: list[str]) -> str | None:
    if not chunks:
        return None
    path = _write_task_artifact(
        run_dir,
        "logs",
        task_id,
        candidate_index,
        "_errors.txt",
        "\n\n".join(chunks),
    )
    return str(path)


def _verification_failure_summary(verification: VerificationResult) -> str:
    lines: list[str] = []
    for case in verification.cases:
        if case.passed:
            continue
        detail = case.message or case.error_type or "verification failed"
        lines.append(
            f"seed={case.seed} shape={list(case.shape)} type={case.error_type or 'unknown'}: {detail}"
        )
    return "\n".join(lines) if lines else "verification failed without case details"


def _candidate_benchmark_summary(benchmark: dict[str, Any] | None) -> dict[str, Any]:
    if not benchmark:
        return {}
    candidate = benchmark.get("candidate") or {}
    eager = benchmark.get("eager") or {}
    compiled = benchmark.get("torch_compile") or {}
    return {
        "candidate_median_ms": candidate.get("median_ms"),
        "eager_median_ms": eager.get("median_ms"),
        "torch_compile_median_ms": compiled.get("median_ms"),
        "speedup_vs_eager": benchmark.get("speedup_vs_eager"),
        "speedup_vs_torch_compile": benchmark.get("speedup_vs_torch_compile"),
        "stable_above_eager": benchmark.get("stable_above_eager"),
        "stable_above_compile": benchmark.get("stable_above_compile"),
        "across_session_median_speedup": benchmark.get("across_session_median_speedup"),
        "across_session_iqr": benchmark.get("across_session_iqr"),
        "compile_error": benchmark.get("compile_error"),
        "benchmark_error": benchmark.get("benchmark_error"),
        "torch_compile_mode": benchmark.get("torch_compile_mode"),
    }


def _candidate_label(
    *,
    policy_passed: bool,
    verification_passed: bool,
    benchmark_summary: dict[str, Any],
    failure_reason: str | None,
) -> str:
    if not policy_passed:
        return "POLICY_FAILED"
    if not verification_passed:
        return "VERIFICATION_FAILED"
    speedup = _float_or_none(benchmark_summary.get("speedup_vs_eager"))
    if speedup is None:
        return "INSUFFICIENT_DATA" if not failure_reason else "BENCHMARK_FAILED"
    if speedup >= 1.0 and benchmark_summary.get("stable_above_eager") is True:
        return "REPEAT_STABLE_WIN"
    if speedup >= 1.0:
        return "UNSTABLE"
    return "BELOW_EAGER"


def _candidate_failure_reason(
    policy: CandidatePolicyResult,
    verification: VerificationResult,
    benchmark_summary: dict[str, Any],
) -> str | None:
    if not policy.passed:
        return policy.rejection_reason or "policy_rejected"
    if not verification.passed:
        return "verification_failed"
    if benchmark_summary.get("benchmark_error"):
        return "benchmark_failed"
    return None


def _candidate_report_lines(candidate_records: list[dict[str, Any]]) -> list[str]:
    total = len(candidate_records)
    source_types = {str(item.get("source_type") or "") for item in candidate_records}
    count_label = "Candidates evaluated" if source_types == {"existing_file"} else "Candidates generated"
    policy_passed = sum(1 for item in candidate_records if item.get("policy_passed"))
    verified = sum(1 for item in candidate_records if item.get("verification_passed"))
    benchmarked = sum(1 for item in candidate_records if item.get("benchmarked"))
    eager_wins = sum(
        1
        for item in candidate_records
        if (_float_or_none((item.get("benchmark_summary") or {}).get("speedup_vs_eager")) or 0.0) >= 1.0
    )
    compile_wins = sum(
        1
        for item in candidate_records
        if (_float_or_none((item.get("benchmark_summary") or {}).get("speedup_vs_torch_compile")) or 0.0) >= 1.0
    )
    label_counts: dict[str, int] = {}
    for item in candidate_records:
        label = str(item.get("candidate_label") or "UNKNOWN")
        label_counts[label] = label_counts.get(label, 0) + 1

    lines = [
        f"- {count_label}: {total}",
        f"- Policy passed/failed: {policy_passed}/{total - policy_passed}",
        f"- Verification passed/failed: {verified}/{total - verified}",
        f"- Benchmarked: {benchmarked}",
        f"- Candidates >= eager: {eager_wins}",
        f"- Candidates >= torch.compile: {compile_wins}",
    ]
    for label, count in sorted(label_counts.items()):
        lines.append(f"- `{label}`: {count}")
    lines.extend(
        [
            "",
            "| Task | Policy | Verified | Label | Speedup vs eager | Speedup vs compile | Failure |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for item in candidate_records:
        summary = item.get("benchmark_summary") or {}
        failure = item.get("failure_reason") or ""
        lines.append(
            "| `{task}` | {policy} | {verified} | `{label}` | {eager} | {compile} | {failure} |".format(
                task=item.get("task_id"),
                policy="yes" if item.get("policy_passed") else "no",
                verified="yes" if item.get("verification_passed") else "no",
                label=item.get("candidate_label") or "UNKNOWN",
                eager=_fmt(summary.get("speedup_vs_eager")),
                compile=_fmt(summary.get("speedup_vs_torch_compile")),
                failure=_short_failure(failure),
            )
        )
    return lines


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(0, (limit - 80) // 2)
    return text[:half] + "\n# ... source truncated for prompt budget ...\n" + text[-half:]


def _new_run_dir(output_dir: str) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = root / stamp
    counter = 1
    while path.exists():
        path = root / f"{stamp}_{counter}"
        counter += 1
    path.mkdir(parents=True)
    return path


def _check_execution_requirements(config: RunConfig, env: Any) -> None:
    if config.execution.disabled_reason:
        raise RuntimeError(f"Config is disabled: {config.execution.disabled_reason}")
    if config.execution.require_cuda and not env.cuda_available:
        raise RuntimeError("Config requires CUDA, but CUDA is unavailable.")
    if config.execution.require_triton and not env.triton_available:
        raise RuntimeError("Config requires Triton, but Triton is unavailable.")
    if config.execution.require_tiny_triton_kernel and not env.tiny_triton_kernel_passed:
        raise RuntimeError("Config requires a passing tiny Triton kernel, but the probe failed.")


def _validate_kernelbench_run_config(
    config: RunConfig,
    kernelbench_cfg: dict[str, Any],
    *,
    provider_mode: str,
    max_tasks: int,
    max_numel_per_input: int | None,
    max_total_input_bytes: int | None,
    max_memory_mb: int | None,
) -> None:
    if max_tasks <= 0:
        raise KernelBenchL1Error("kernelbench.max_tasks must be positive")
    for name, value in (
        ("max_numel_per_input", max_numel_per_input),
        ("max_total_input_bytes", max_total_input_bytes),
        ("skip_if_estimated_memory_gt_mb", max_memory_mb),
    ):
        if value is not None and value <= 0:
            raise KernelBenchL1Error(f"kernelbench.{name} must be positive or null")

    if provider_mode not in {"gemini", "gemini_repair"}:
        return
    if int(kernelbench_cfg.get("candidates_per_task", 1)) != 1:
        raise KernelBenchL1Error(
            "KernelBench Gemini provider currently supports exactly one candidate per task"
        )
    if provider_mode == "gemini_repair" and int(
        kernelbench_cfg.get("max_repair_candidates_per_task", 1)
    ) != 1:
        raise KernelBenchL1Error(
            "KernelBench Gemini repair provider currently supports exactly one repair per task"
        )
    if config.agent.max_attempts != 1 or config.agent.candidates_per_attempt != 1:
        raise KernelBenchL1Error(
            "KernelBench Gemini provider requires agent.max_attempts=1 and "
            "agent.candidates_per_attempt=1"
        )
    if config.agent.performance_search.enabled:
        raise KernelBenchL1Error("KernelBench Gemini provider does not support performance search")
    credential_path = _serialized_credential_path(
        {
            "api_key": config.agent.api_key,
            "extra_headers": config.agent.extra_headers,
            "extra_body": config.agent.extra_body,
            "backend_options": config.agent.backend_options,
        }
    )
    if credential_path:
        raise KernelBenchL1Error(
            "KernelBench model credentials must use agent.api_key_env, not serialized field "
            + credential_path
        )
    normalized_backend = config.agent.backend.replace("_", "-").lower()
    if normalized_backend in {"openai", "openai-compatible", "openai-responses", "responses"}:
        env_name = config.agent.api_key_env
        if not env_name or not os.environ.get(env_name):
            raise KernelBenchL1Error(
                f"Missing API key environment variable: export {env_name or 'GEMINI_API_KEY'}=<key>"
            )


def _serialized_credential_path(value: Any, *, path: str = "agent") -> str | None:
    """Return the first non-empty credential-like config path, excluding env names."""

    credential_keys = {
        "api_key",
        "authorization",
        "access_token",
        "refresh_token",
        "bearer_token",
        "password",
        "secret",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            child_path = f"{path}.{key}"
            is_credential = (
                normalized in credential_keys
                or normalized.endswith("_api_key")
                or normalized.endswith("_secret")
                or normalized.endswith("_password")
                or normalized.endswith("_token")
            )
            if is_credential and not normalized.endswith("_env") and item is not None and item != "":
                return child_path
            nested = _serialized_credential_path(item, path=child_path)
            if nested:
                return nested
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            nested = _serialized_credential_path(item, path=f"{path}[{index}]")
            if nested:
                return nested
    return None


def _select_device(config: RunConfig) -> str:
    if config.benchmark.device != "auto":
        return config.benchmark.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _benchmark_to_dict(result: BenchmarkResult) -> dict[str, Any]:
    return _to_jsonable(dataclasses.asdict(result))


def _benchmark_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    eager = benchmark.get("eager") or {}
    candidate = benchmark.get("candidate") or {}
    compiled = benchmark.get("torch_compile") or {}
    return {
        "shape": benchmark.get("shape"),
        "dtype": benchmark.get("dtype"),
        "device": benchmark.get("device"),
        "eager_median_ms": eager.get("median_ms"),
        "torch_compile_median_ms": compiled.get("median_ms"),
        "torch_compile_mode": benchmark.get("torch_compile_mode"),
        "timing_mode": benchmark.get("timing_mode"),
        "warmup": benchmark.get("warmup"),
        "repeat": benchmark.get("repeats"),
        "independent_sessions": benchmark.get("independent_sessions"),
        "cache_flush_enabled": benchmark.get("cache_flush_enabled"),
        "cache_flush_performed": benchmark.get("cache_flush_performed"),
        "eager_ms_summary": benchmark.get("eager_ms_summary"),
        "torch_compile_ms_summary": benchmark.get("torch_compile_ms_summary"),
        "speedup_vs_compile": benchmark.get("speedup_vs_torch_compile"),
        "compile_time_ms": benchmark.get("compile_time_ms"),
        "measurement_warnings": benchmark.get("measurement_warnings") or [],
        "benchmark_error": benchmark.get("benchmark_error"),
        "compile_error": benchmark.get("compile_error"),
        "reference_self_median_ms": candidate.get("median_ms"),
    }


def _output_summary(output: Any) -> dict[str, Any]:
    if isinstance(output, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(output.shape),
            "dtype": str(output.dtype).replace("torch.", ""),
            "device": str(output.device),
        }
    if isinstance(output, (list, tuple)):
        return {"type": type(output).__name__, "length": len(output)}
    return {"type": type(output).__name__}


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.device):
        return str(value)
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _short_failure(value: Any) -> str:
    if not value:
        return ""
    text = str(value).splitlines()[-1] if "\n" in str(value) else str(value)
    text = text.replace("|", "\\|")
    return text[:220] + ("..." if len(text) > 220 else "")
