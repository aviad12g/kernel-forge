"""End-to-end task runner."""

from __future__ import annotations

import dataclasses
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from openkernelforge.agents.backends import create_backend
from openkernelforge.agents.dummy_agent import DummyAgent
from openkernelforge.agents.llm_agent import LLMAgent
from openkernelforge.config import AgentConfig, RunConfig, load_config, save_config
from openkernelforge.harness.benchmarker import benchmark_task
from openkernelforge.harness.policy import CandidatePolicyResult, check_candidate_policy
from openkernelforge.harness.sandbox import (
    CandidateLoadError,
    LoadedCandidate,
    load_candidate_from_path,
    resolve_task_artifact_dir,
    unload_candidate,
    write_candidate_source,
)
from openkernelforge.harness.template_preservation import check_template_preservation
from openkernelforge.harness.verifier import VerificationResult, verify_candidate
from openkernelforge.reports.summarize import write_summary
from openkernelforge.tasks.base import KernelTask, Shape
from openkernelforge.tasks.simple_tasks import get_task
from openkernelforge.templates.template_agent import TemplateAgent
from openkernelforge.utils.env_probe import EnvironmentProbeResult, probe_environment
from openkernelforge.utils.gpu import resolve_device


def run_from_config(config_or_path: RunConfig | str | Path) -> Path:
    run_started = datetime.now(timezone.utc)
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    run_dir = _make_run_dir(config.output_dir)
    (run_dir / "candidates").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (run_dir / "responses").mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    environment = probe_environment()
    _write_environment_probe(run_dir, environment)
    _enforce_execution_requirements(config, environment)

    records: list[dict[str, Any]] = []
    results_path = run_dir / "results.jsonl"
    dummy_agent = DummyAgent()
    template_agent = (
        TemplateAgent(
            template_family=config.agent.template_family,
            template_variants=config.agent.template_variants,
        )
        if config.agent.type == "template"
        else None
    )
    llm_agent = _build_llm_agent(config.agent) if config.agent.type == "llm" else None
    verification_device = resolve_device(config.verification.device)
    benchmark_device = resolve_device(config.benchmark.device)

    with results_path.open("w", encoding="utf-8") as results_file:
        for task_id in config.tasks:
            task = get_task(task_id)
            if config.agent.type == "dummy":
                record = _run_dummy_task(
                    task,
                    run_dir=run_dir,
                    config=config,
                    agent=dummy_agent,
                    verification_device=verification_device,
                    benchmark_device=benchmark_device,
                )
            elif config.agent.type == "template":
                if template_agent is None:
                    raise RuntimeError("Template agent was not initialized")
                record = _run_template_task(
                    task,
                    run_dir=run_dir,
                    config=config,
                    agent=template_agent,
                    verification_device=verification_device,
                    benchmark_device=benchmark_device,
                )
            elif config.agent.type == "llm":
                if llm_agent is None:
                    raise RuntimeError("LLM agent was not initialized")
                record = _run_llm_task(
                    task,
                    run_dir=run_dir,
                    config=config,
                    agent=llm_agent,
                    verification_device=verification_device,
                    benchmark_device=benchmark_device,
                )
            else:
                raise ValueError(f"Unknown agent type: {config.agent.type}")

            for candidate_record in record.get("candidate_records", []):
                results_file.write(json.dumps(candidate_record) + "\n")
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()
            records.append(record)

    write_summary(run_dir, records)
    run_completed = datetime.now(timezone.utc)
    _write_run_metadata(
        run_dir,
        config=config,
        environment=environment,
        started_at=run_started,
        completed_at=run_completed,
        records=records,
    )
    if config.agent.performance_search.enabled:
        from openkernelforge.reports.performance_search import write_performance_search_report

        write_performance_search_report(run_dir)
        if config.agent.performance_search.mode == "template_copy":
            from openkernelforge.reports.template_copy import write_template_copy_report

            write_template_copy_report(run_dir)
    if config.agent.type == "template":
        from openkernelforge.reports.template_report import write_template_autotune_report

        if template_agent is not None:
            from openkernelforge.reports.skipped_variants import write_skipped_variants_artifacts

            write_skipped_variants_artifacts(run_dir, template_agent.skipped_variants_by_task)
        write_template_autotune_report(run_dir)
        if config.agent.template_family == "fused8":
            from openkernelforge.reports.fused8 import write_fused8_report

            write_fused8_report(run_dir)
        if config.agent.template_variants.get("generation_stage") in {
            "template_focused_sweep",
            "template_focused_clean",
        }:
            from openkernelforge.reports.focused_sweep import write_focused_sweep_report

            write_focused_sweep_report(run_dir)
    return run_dir


def _build_llm_agent(agent_config: AgentConfig) -> LLMAgent:
    backend = create_backend(agent_config)
    return LLMAgent(
        backend,
        backend_name=agent_config.backend,
        max_attempts=agent_config.max_attempts,
        allow_torch_fallback=agent_config.allow_torch_fallback,
        temperature=agent_config.temperature,
        candidates_per_attempt=agent_config.candidates_per_attempt,
        prompt_version=agent_config.prompt_version,
        repair_prompt_version=agent_config.repair_prompt_version,
        performance_prompt_version=agent_config.performance_prompt_version,
    )


def _run_dummy_task(
    task: KernelTask,
    *,
    run_dir: Path,
    config: RunConfig,
    agent: DummyAgent,
    verification_device: torch.device,
    benchmark_device: torch.device,
) -> dict[str, Any]:
    candidate_index = 0
    candidate_id = f"candidate_{candidate_index:03d}"
    candidate_spec = agent.generate(task, device=str(verification_device))
    candidate_path = write_candidate_source(
        run_dir,
        task.task_id,
        candidate_index,
        candidate_spec.source,
    )

    policy = check_candidate_policy(
        candidate_spec.source,
        allow_torch_fallback=config.agent.allow_torch_fallback,
        require_triton=not config.agent.allow_torch_fallback,
    )
    if policy.passed:
        loaded, verification, error_chunks = _load_and_verify(
            task,
            candidate_path,
            candidate_spec.name,
            config,
            verification_device,
        )
    else:
        loaded = None
        verification = _policy_rejected_verification(task, candidate_spec.name, policy)
        error_chunks = ["Policy rejected candidate:\n" + str(policy.rejection_reason)]
    benchmarks: list[Any] = []
    if verification.passed and config.benchmark.enabled and loaded is not None:
        benchmarks, benchmark_errors = _benchmark_candidate(
            task,
            loaded,
            candidate_spec.name,
            config,
            benchmark_device,
        )
        error_chunks.extend(benchmark_errors)

    error_log_value = _write_error_log(run_dir, task.task_id, candidate_index, error_chunks)
    attempt = {
        "attempt_index": candidate_index,
        "candidate_index": 0,
        "candidate_id": candidate_id,
        "candidate_name": candidate_spec.name,
        "prompt_path": None,
        "response_path": None,
        "candidate_path": str(candidate_path),
        "extraction": None,
        "verification": _to_jsonable(verification),
        "policy": _to_jsonable(policy),
        "benchmarks": [_to_jsonable(benchmark) for benchmark in benchmarks],
        "failure_reason": _failure_reason(verification, policy=policy),
        "error_log_path": error_log_value,
        "generation_stage": "initial",
    }
    candidate_record = _candidate_json_record(
        task=task,
        config=config,
        attempt=attempt,
        backend=str(candidate_spec.metadata.get("backend", "torch")),
        model=None,
        selected_best=True,
    )
    if loaded is not None:
        unload_candidate(loaded)

    return {
        "record_type": "task_summary",
        "task_id": task.task_id,
        "task_name": task.name,
        "agent_type": "dummy",
        "backend": candidate_spec.metadata.get("backend", "torch"),
        "candidate_id": candidate_id,
        "candidate_name": candidate_spec.name,
        "candidate_path": str(candidate_path),
        "candidate_metadata": candidate_spec.metadata,
        "verification": _to_jsonable(verification),
        "benchmarks": [_to_jsonable(benchmark) for benchmark in benchmarks],
        "attempts": [attempt],
        "candidate_records": [candidate_record],
        "error_log_path": error_log_value,
    }


def _run_template_task(
    task: KernelTask,
    *,
    run_dir: Path,
    config: RunConfig,
    agent: TemplateAgent,
    verification_device: torch.device,
    benchmark_device: torch.device,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    for candidate_index, candidate_spec in enumerate(agent.generate_all(task)):
        candidate_path = write_candidate_source(
            run_dir,
            task.task_id,
            candidate_index,
            candidate_spec.source,
        )
        error_chunks: list[str] = []
        loaded: LoadedCandidate | None = None
        policy = check_candidate_policy(
            candidate_spec.source,
            allow_torch_fallback=config.agent.allow_torch_fallback,
            require_triton=not config.agent.allow_torch_fallback,
        )
        if policy.passed:
            loaded, verification, error_chunks = _load_and_verify(
                task,
                candidate_path,
                candidate_spec.name,
                config,
                verification_device,
            )
        else:
            verification = _policy_rejected_verification(task, candidate_spec.name, policy)
            error_chunks.append("Policy rejected candidate:\n" + str(policy.rejection_reason))

        benchmarks: list[Any] = []
        if verification.passed and config.benchmark.enabled and loaded is not None:
            benchmarks, benchmark_errors = _benchmark_candidate(
                task,
                loaded,
                candidate_spec.name,
                config,
                benchmark_device,
            )
            error_chunks.extend(benchmark_errors)

        error_log_value = _write_error_log(run_dir, task.task_id, candidate_index, error_chunks)
        generation_metadata = {
            **candidate_spec.metadata,
            "template_family": agent.template_family,
        }
        attempt = {
            "attempt_index": 0,
            "candidate_index": candidate_index,
            "candidate_id": f"candidate_{candidate_index:03d}",
            "candidate_name": candidate_spec.name,
            "prompt_path": None,
            "response_path": None,
            "candidate_path": str(candidate_path),
            "extraction": None,
            "verification": _to_jsonable(verification),
            "policy": _to_jsonable(policy),
            "benchmarks": [_to_jsonable(benchmark) for benchmark in benchmarks],
            "failure_reason": _failure_reason(verification, policy=policy),
            "error_log_path": error_log_value,
            "generation_metadata": _to_jsonable(generation_metadata),
            "generation_stage": generation_metadata.get("generation_stage", "template_baseline"),
            "task_family": generation_metadata.get("task_family"),
            "template_family": generation_metadata.get("template_family"),
            "template_id": generation_metadata.get("template_id"),
            "block_size": generation_metadata.get("block_size"),
            "reduction_block_size": generation_metadata.get("reduction_block_size"),
            "reduction_axis": generation_metadata.get("reduction_axis"),
            "num_warps": generation_metadata.get("num_warps"),
            "num_stages": generation_metadata.get("num_stages"),
            "contiguous_policy": generation_metadata.get("contiguous_policy"),
            "output_allocation_policy": generation_metadata.get("output_allocation_policy"),
            "shape_specialized": generation_metadata.get("shape_specialized"),
            "feature_dim_mode": generation_metadata.get("feature_dim_mode"),
            "n_elements_mode": generation_metadata.get("n_elements_mode"),
            "total_possible_variants": generation_metadata.get("total_possible_variants"),
            "valid_possible_variants": generation_metadata.get("valid_possible_variants"),
            "generated_valid_variants": generation_metadata.get("generated_valid_variants"),
            "skipped_invalid_variants": generation_metadata.get("skipped_invalid_variants"),
            "skipped_reasons": generation_metadata.get("skipped_reasons"),
            "variant_validation": generation_metadata.get("variant_validation"),
            "actually_generated_variants": generation_metadata.get("actually_generated_variants"),
            "grid_sampling": generation_metadata.get("grid_sampling"),
            "sort_order": generation_metadata.get("sort_order"),
            "grid_was_capped": generation_metadata.get("grid_was_capped"),
            "focused_seed_run": generation_metadata.get("focused_seed_run"),
            "focused_seed_candidate_path": generation_metadata.get("focused_seed_candidate_path"),
        }
        attempts.append(attempt)
        candidate_records.append(
            _candidate_json_record(
                task=task,
                config=config,
                attempt=attempt,
                backend=str(candidate_spec.metadata.get("backend", "triton_template")),
                model=None,
                selected_best=False,
            )
        )
        if loaded is not None:
            unload_candidate(loaded)

    final_attempt = _select_best_attempt(attempts)
    _mark_selected_best(candidate_records, final_attempt)
    return {
        "record_type": "task_summary",
        "task_id": task.task_id,
        "task_name": task.name,
        "agent_type": "template",
        "backend": "triton_template",
        "candidate_id": final_attempt.get("candidate_id"),
        "candidate_name": final_attempt.get("candidate_name"),
        "candidate_path": final_attempt.get("candidate_path"),
        "candidate_metadata": {
            "agent": "template",
            "template_family": config.agent.template_family,
            "template_variants": config.agent.template_variants,
        },
        "verification": final_attempt.get("verification"),
        "benchmarks": final_attempt.get("benchmarks", []),
        "attempts": attempts,
        "candidate_records": candidate_records,
        "error_log_path": final_attempt.get("error_log_path"),
    }


def _run_llm_task(
    task: KernelTask,
    *,
    run_dir: Path,
    config: RunConfig,
    agent: LLMAgent,
    verification_device: torch.device,
    benchmark_device: torch.device,
) -> dict[str, Any]:
    if (
        config.agent.performance_search.enabled
        and config.agent.performance_search.mode == "template_copy"
    ):
        return _run_template_copy_task(
            task=task,
            run_dir=run_dir,
            config=config,
            agent=agent,
            verification_device=verification_device,
            benchmark_device=benchmark_device,
        )

    attempts: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    original_prompt: str | None = None
    previous_candidate = ""
    previous_verification: VerificationResult | None = None
    previous_failure: str | None = None
    global_candidate_index = 0

    for attempt_index in range(agent.max_attempts):
        attempt_entries: list[dict[str, Any]] = []
        for candidate_index in range(agent.candidates_per_attempt):
            if attempt_index == 0:
                generation = agent.generate_initial(
                    task,
                    attempt_index=attempt_index,
                    candidate_index=candidate_index,
                )
                original_prompt = original_prompt or generation.prompt
            else:
                generation = agent.generate_repair(
                    task,
                    attempt_index=attempt_index,
                    candidate_index=candidate_index,
                    original_task_prompt=original_prompt or agent.initial_prompt(task),
                    previous_candidate=previous_candidate,
                    verification=previous_verification,
                    extra_failure=previous_failure,
                )

            prompt_path = _write_text_artifact(
                run_dir,
                "prompts",
                task.task_id,
                global_candidate_index,
                "_prompt.txt",
                generation.prompt,
            )
            response_path = _write_text_artifact(
                run_dir,
                "responses",
                task.task_id,
                global_candidate_index,
                "_response.txt",
                generation.raw_response,
            )

            candidate_source = generation.extraction.code
            extraction_error = generation.extraction.error
            if candidate_source is None:
                candidate_source = "# Code extraction failed. See the raw response artifact.\n"
            candidate_path = write_candidate_source(
                run_dir,
                task.task_id,
                global_candidate_index,
                candidate_source,
            )

            error_chunks: list[str] = []
            loaded: LoadedCandidate | None = None
            if extraction_error:
                error_chunks.append("Code extraction error:\n" + extraction_error)
                verification = VerificationResult(
                    task_id=task.task_id,
                    candidate_name=generation.candidate_name,
                    passed=False,
                    error=extraction_error,
                )
                policy = CandidatePolicyResult(
                    passed=False,
                    rejection_reason="code_extraction_failed",
                )
            else:
                policy = check_candidate_policy(
                    candidate_source,
                    allow_torch_fallback=config.agent.allow_torch_fallback,
                    require_triton=not config.agent.allow_torch_fallback,
                )
                if policy.passed:
                    loaded, verification, error_chunks = _load_and_verify(
                        task,
                        candidate_path,
                        generation.candidate_name,
                        config,
                        verification_device,
                    )
                else:
                    verification = _policy_rejected_verification(
                        task,
                        generation.candidate_name,
                        policy,
                    )
                    error_chunks.append("Policy rejected candidate:\n" + str(policy.rejection_reason))

            benchmarks: list[Any] = []
            should_benchmark = (
                verification.passed
                and config.benchmark.enabled
                and loaded is not None
                and (
                    config.agent.benchmark_all_correct
                    or not any(record.get("verification_passed") for record in candidate_records)
                )
            )
            if should_benchmark:
                assert loaded is not None
                benchmarks, benchmark_errors = _benchmark_candidate(
                    task,
                    loaded,
                    generation.candidate_name,
                    config,
                    benchmark_device,
                )
                error_chunks.extend(benchmark_errors)

            error_log_value = _write_error_log(
                run_dir, task.task_id, global_candidate_index, error_chunks
            )
            attempt = {
                "attempt_index": attempt_index,
                "candidate_index": candidate_index,
                "candidate_id": f"candidate_{global_candidate_index:03d}",
                "candidate_name": generation.candidate_name,
                "prompt_path": str(prompt_path),
                "response_path": str(response_path),
                "candidate_path": str(candidate_path),
                "extraction": _to_jsonable(generation.extraction),
                "verification": _to_jsonable(verification),
                "policy": _to_jsonable(policy),
                "benchmarks": [_to_jsonable(benchmark) for benchmark in benchmarks],
                "failure_reason": _failure_reason(
                    verification,
                    extraction_error=extraction_error,
                    policy=policy,
                ),
                "error_log_path": error_log_value,
                "generation_metadata": _to_jsonable(generation.metadata),
                "generation_stage": generation.metadata.get("stage", "initial"),
            }
            attempts.append(attempt)
            candidate_record = _candidate_json_record(
                task=task,
                config=config,
                attempt=attempt,
                backend=config.agent.backend,
                model=config.agent.model,
                selected_best=False,
            )
            candidate_records.append(candidate_record)
            attempt_entries.append(
                {
                    "attempt": attempt,
                    "candidate_record": candidate_record,
                    "candidate_source": generation.extraction.code or generation.raw_response,
                    "verification": verification,
                    "error_chunks": error_chunks,
                }
            )
            if loaded is not None:
                unload_candidate(loaded)
            global_candidate_index += 1

        correct_entries = [
            entry
            for entry in attempt_entries
            if entry["attempt"].get("verification", {}).get("passed")
        ]
        if correct_entries and config.agent.stop_after_first_correct:
            break

        repair_entry = _select_repair_entry(
            attempt_entries,
            allow_slow_correct_repair=not config.agent.performance_search.enabled,
        )
        if repair_entry is None:
            break
        previous_candidate = str(repair_entry["candidate_source"])
        previous_verification = repair_entry["verification"]
        previous_failure = (
            "\n\n".join(repair_entry["error_chunks"])
            if repair_entry["error_chunks"]
            else repair_entry["attempt"].get("failure_reason")
        )
        performance_feedback = _performance_feedback(repair_entry["attempt"])
        if performance_feedback:
            previous_failure = (
                f"{previous_failure}\n\n{performance_feedback}"
                if previous_failure
                else performance_feedback
            )

    initial_best_attempt = _select_best_attempt(attempts)
    if config.agent.performance_search.enabled:
        global_candidate_index = _run_performance_search(
            task=task,
            run_dir=run_dir,
            config=config,
            agent=agent,
            verification_device=verification_device,
            benchmark_device=benchmark_device,
            attempts=attempts,
            candidate_records=candidate_records,
            starting_candidate_index=global_candidate_index,
            initial_best_attempt=initial_best_attempt,
        )

    final_attempt = _select_best_attempt(attempts)
    _mark_selected_best(candidate_records, final_attempt)
    _annotate_performance_search_outcome(
        attempts=attempts,
        candidate_records=candidate_records,
        initial_best_attempt=initial_best_attempt,
        final_attempt=final_attempt,
        config=config,
    )
    return {
        "record_type": "task_summary",
        "task_id": task.task_id,
        "task_name": task.name,
        "agent_type": "llm",
        "backend": config.agent.backend,
        "candidate_id": final_attempt.get("candidate_id"),
        "candidate_name": final_attempt.get("candidate_name"),
        "candidate_path": final_attempt.get("candidate_path"),
        "candidate_metadata": {
            "agent": "llm",
            "backend": config.agent.backend,
            "max_attempts": agent.max_attempts,
            "allow_torch_fallback": agent.allow_torch_fallback,
            "prompt_version": agent.prompt_version,
            "repair_prompt_version": agent.repair_prompt_version,
            "performance_prompt_version": agent.performance_prompt_version,
            "performance_search": _to_jsonable(config.agent.performance_search),
        },
        "verification": final_attempt.get("verification"),
        "benchmarks": final_attempt.get("benchmarks", []),
        "attempts": attempts,
        "candidate_records": candidate_records,
        "error_log_path": final_attempt.get("error_log_path"),
    }


def _run_template_copy_task(
    *,
    task: KernelTask,
    run_dir: Path,
    config: RunConfig,
    agent: LLMAgent,
    verification_device: torch.device,
    benchmark_device: torch.device,
) -> dict[str, Any]:
    search_config = config.agent.performance_search
    template_context = _best_template_context(task.task_id, config)
    if not template_context:
        raise RuntimeError(f"No template context available for task {task.task_id}")
    template_code = str(template_context.get("candidate_code") or "")
    template_summary = template_context.get("benchmark_summary") or {}
    settings = _template_copy_settings(config)
    attempts: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    global_candidate_index = 0

    for setting_index, setting in enumerate(settings):
        for setting_candidate_index in range(max(search_config.candidates_per_setting, 1)):
            generation = agent.generate_template_copy_candidate(
                task,
                attempt_index=setting_index,
                candidate_index=setting_candidate_index,
                template_code=template_code,
                template_benchmark_summary=template_summary,
                requested_parameters=setting,
                heuristic_flags=_static_performance_flags(task.task_id, template_code),
                template_context=template_context,
            )
            prompt_path = _write_text_artifact(
                run_dir,
                "prompts",
                task.task_id,
                global_candidate_index,
                "_prompt.txt",
                generation.prompt,
            )
            response_path = _write_text_artifact(
                run_dir,
                "responses",
                task.task_id,
                global_candidate_index,
                "_response.txt",
                generation.raw_response,
            )
            candidate_source = generation.extraction.code
            extraction_error = generation.extraction.error
            if candidate_source is None:
                candidate_source = "# Code extraction failed. See the raw response artifact.\n"
            candidate_path = write_candidate_source(
                run_dir,
                task.task_id,
                global_candidate_index,
                candidate_source,
            )

            error_chunks: list[str] = []
            loaded: LoadedCandidate | None = None
            preservation = None
            if extraction_error:
                error_chunks.append("Code extraction error:\n" + extraction_error)
                verification = VerificationResult(
                    task_id=task.task_id,
                    candidate_name=generation.candidate_name,
                    passed=False,
                    error=extraction_error,
                )
                policy = CandidatePolicyResult(
                    passed=False,
                    rejection_reason="code_extraction_failed",
                )
            else:
                policy = check_candidate_policy(
                    candidate_source,
                    allow_torch_fallback=config.agent.allow_torch_fallback,
                    require_triton=not config.agent.allow_torch_fallback,
                )
                if policy.passed:
                    preservation = check_template_preservation(
                        candidate_source,
                        template_code,
                        task_id=task.task_id,
                        reject_if_score_below=(
                            config.agent.template_copy.reject_if_preservation_score_below
                        ),
                        reject_fallbacks=config.agent.template_copy.reject_fallbacks,
                        reject_forbidden_torch_ops=(
                            config.agent.template_copy.reject_forbidden_torch_ops
                        ),
                    )
                    if preservation.passed:
                        loaded, verification, error_chunks = _load_and_verify(
                            task,
                            candidate_path,
                            generation.candidate_name,
                            config,
                            verification_device,
                        )
                    else:
                        verification = VerificationResult(
                            task_id=task.task_id,
                            candidate_name=generation.candidate_name,
                            passed=False,
                            error=f"template_preservation_rejected:{preservation.rejection_reason}",
                        )
                        error_chunks.append(
                            "Template preservation rejected candidate:\n"
                            + str(preservation.rejection_reason)
                            + "\n"
                            + "\n".join(preservation.warnings)
                        )
                else:
                    verification = _policy_rejected_verification(
                        task,
                        generation.candidate_name,
                        policy,
                    )
                    error_chunks.append("Policy rejected candidate:\n" + str(policy.rejection_reason))

            benchmarks: list[Any] = []
            if verification.passed and config.benchmark.enabled and loaded is not None:
                benchmarks, benchmark_errors = _benchmark_candidate(
                    task,
                    loaded,
                    generation.candidate_name,
                    config,
                    benchmark_device,
                )
                error_chunks.extend(benchmark_errors)

            candidate_summary = _benchmark_summary([_to_jsonable(item) for item in benchmarks]) or {}
            template_speedup = (template_summary or {}).get("speedup_vs_eager")
            candidate_speedup = candidate_summary.get("speedup_vs_eager")
            delta_vs_template = (
                float(candidate_speedup) - float(template_speedup)
                if candidate_speedup is not None and template_speedup is not None
                else None
            )
            preservation_dict = preservation.to_dict() if preservation is not None else None
            checks = (preservation_dict or {}).get("checks") or {}
            error_log_value = _write_error_log(
                run_dir, task.task_id, global_candidate_index, error_chunks
            )
            generation_metadata = {
                **dict(generation.metadata),
                "requested_block_size": setting.get("block_size"),
                "requested_num_warps": setting.get("num_warps"),
                "requested_contiguous_policy": setting.get("contiguous_policy"),
                "template_source_path": template_context.get("candidate_path"),
                "copied_from_template_id": template_context.get("template_id"),
            }
            attempt = {
                "attempt_index": setting_index,
                "candidate_index": setting_candidate_index,
                "candidate_id": f"candidate_{global_candidate_index:03d}",
                "candidate_name": generation.candidate_name,
                "prompt_path": str(prompt_path),
                "response_path": str(response_path),
                "candidate_path": str(candidate_path),
                "extraction": _to_jsonable(generation.extraction),
                "verification": _to_jsonable(verification),
                "policy": _to_jsonable(policy),
                "benchmarks": [_to_jsonable(benchmark) for benchmark in benchmarks],
                "failure_reason": (
                    "template_preservation_rejected"
                    if preservation is not None and not preservation.passed
                    else _failure_reason(
                        verification,
                        extraction_error=extraction_error,
                        policy=policy,
                    )
                ),
                "error_log_path": error_log_value,
                "generation_metadata": _to_jsonable(generation_metadata),
                "generation_stage": "template_copy",
                "search_round": setting_index + 1,
                "template_source_path": template_context.get("candidate_path"),
                "copied_from_template_id": template_context.get("template_id"),
                "requested_block_size": setting.get("block_size"),
                "requested_num_warps": setting.get("num_warps"),
                "requested_contiguous_policy": setting.get("contiguous_policy"),
                "template_preservation": preservation_dict,
                "preserved_template_structure_score": (
                    preservation.score if preservation is not None else None
                ),
                "extra_torch_ops_detected": bool(checks.get("forbidden_torch_ops")),
                "fallback_detected": bool(checks.get("fallback_detected")),
                "source_template_benchmark": template_summary,
                "source_template_speedup_vs_eager": template_speedup,
                "delta_vs_source_template": delta_vs_template,
                "parent_candidate_path": template_context.get("candidate_path"),
                "parent_candidate_id": template_context.get("template_id"),
                "parent_speedup_vs_eager": template_speedup,
                "improved_over_parent": (
                    bool(delta_vs_template is not None and delta_vs_template > 0)
                ),
                "target_reached": _target_reached_from_summary(candidate_summary, config),
            }
            attempts.append(attempt)
            candidate_records.append(
                _candidate_json_record(
                    task=task,
                    config=config,
                    attempt=attempt,
                    backend=config.agent.backend,
                    model=config.agent.model,
                    selected_best=False,
                )
            )
            if loaded is not None:
                unload_candidate(loaded)
            global_candidate_index += 1

    final_attempt = _select_best_attempt(attempts)
    _mark_selected_best(candidate_records, final_attempt)
    _annotate_performance_search_outcome(
        attempts=attempts,
        candidate_records=candidate_records,
        initial_best_attempt=attempts[0],
        final_attempt=final_attempt,
        config=config,
    )
    return {
        "record_type": "task_summary",
        "task_id": task.task_id,
        "task_name": task.name,
        "agent_type": "llm",
        "backend": config.agent.backend,
        "candidate_id": final_attempt.get("candidate_id"),
        "candidate_name": final_attempt.get("candidate_name"),
        "candidate_path": final_attempt.get("candidate_path"),
        "candidate_metadata": {
            "agent": "llm",
            "backend": config.agent.backend,
            "prompt_version": agent.prompt_version,
            "repair_prompt_version": agent.repair_prompt_version,
            "performance_prompt_version": agent.performance_prompt_version,
            "performance_search": _to_jsonable(config.agent.performance_search),
            "template_copy": _to_jsonable(config.agent.template_copy),
        },
        "verification": final_attempt.get("verification"),
        "benchmarks": final_attempt.get("benchmarks", []),
        "attempts": attempts,
        "candidate_records": candidate_records,
        "error_log_path": final_attempt.get("error_log_path"),
    }


def _run_performance_search(
    *,
    task: KernelTask,
    run_dir: Path,
    config: RunConfig,
    agent: LLMAgent,
    verification_device: torch.device,
    benchmark_device: torch.device,
    attempts: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    starting_candidate_index: int,
    initial_best_attempt: dict[str, Any],
) -> int:
    search_config = config.agent.performance_search
    parents = _performance_search_parents(attempts, initial_best_attempt, config)
    global_candidate_index = starting_candidate_index
    for root_parent in parents:
        if not _needs_performance_search(root_parent, config):
            continue
        best_attempt = root_parent
        best_source = _read_text_path(best_attempt.get("candidate_path"))
        for search_round in range(1, search_config.max_rounds + 1):
            if _target_reached(best_attempt, config):
                break
            parent_attempt = best_attempt
            parent_source = best_source or _read_text_path(parent_attempt.get("candidate_path"))
            parent_summary = _benchmark_summary(parent_attempt.get("benchmarks") or []) or {}
            parent_speedup = parent_summary.get("speedup_vs_eager")
            heuristic_flags = _static_performance_flags(task.task_id, parent_source)
            template_context = _best_template_context(task.task_id, config)
            for candidate_index in range(search_config.candidates_per_round):
                generation = agent.generate_performance_candidate(
                    task,
                    attempt_index=agent.max_attempts + search_round - 1,
                    candidate_index=candidate_index,
                    previous_candidate=parent_source,
                    benchmark_summary=parent_summary,
                    heuristic_flags=heuristic_flags,
                    template_context=template_context,
                )
                prompt_path = _write_text_artifact(
                    run_dir,
                    "prompts",
                    task.task_id,
                    global_candidate_index,
                    "_prompt.txt",
                    generation.prompt,
                )
                response_path = _write_text_artifact(
                    run_dir,
                    "responses",
                    task.task_id,
                    global_candidate_index,
                    "_response.txt",
                    generation.raw_response,
                )

                candidate_source = generation.extraction.code
                extraction_error = generation.extraction.error
                if candidate_source is None:
                    candidate_source = "# Code extraction failed. See the raw response artifact.\n"
                candidate_path = write_candidate_source(
                    run_dir,
                    task.task_id,
                    global_candidate_index,
                    candidate_source,
                )

                error_chunks: list[str] = []
                loaded: LoadedCandidate | None = None
                if extraction_error:
                    error_chunks.append("Code extraction error:\n" + extraction_error)
                    verification = VerificationResult(
                        task_id=task.task_id,
                        candidate_name=generation.candidate_name,
                        passed=False,
                        error=extraction_error,
                    )
                    policy = CandidatePolicyResult(
                        passed=False,
                        rejection_reason="code_extraction_failed",
                    )
                else:
                    policy = check_candidate_policy(
                        candidate_source,
                        allow_torch_fallback=config.agent.allow_torch_fallback,
                        require_triton=not config.agent.allow_torch_fallback,
                    )
                    if policy.passed:
                        loaded, verification, error_chunks = _load_and_verify(
                            task,
                            candidate_path,
                            generation.candidate_name,
                            config,
                            verification_device,
                        )
                    else:
                        verification = _policy_rejected_verification(
                            task,
                            generation.candidate_name,
                            policy,
                        )
                        error_chunks.append(
                            "Policy rejected candidate:\n" + str(policy.rejection_reason)
                        )

                benchmarks: list[Any] = []
                if verification.passed and config.benchmark.enabled and loaded is not None:
                    benchmarks, benchmark_errors = _benchmark_candidate(
                        task,
                        loaded,
                        generation.candidate_name,
                        config,
                        benchmark_device,
                    )
                    error_chunks.extend(benchmark_errors)

                child_summary = _benchmark_summary([_to_jsonable(item) for item in benchmarks]) or {}
                child_speedup = child_summary.get("speedup_vs_eager")
                improved = _speedup_greater(child_speedup, parent_speedup)
                error_log_value = _write_error_log(
                    run_dir, task.task_id, global_candidate_index, error_chunks
                )
                attempt = {
                    "attempt_index": agent.max_attempts + search_round - 1,
                    "candidate_index": candidate_index,
                    "candidate_id": f"candidate_{global_candidate_index:03d}",
                    "candidate_name": generation.candidate_name,
                    "prompt_path": str(prompt_path),
                    "response_path": str(response_path),
                    "candidate_path": str(candidate_path),
                    "extraction": _to_jsonable(generation.extraction),
                    "verification": _to_jsonable(verification),
                    "policy": _to_jsonable(policy),
                    "benchmarks": [_to_jsonable(benchmark) for benchmark in benchmarks],
                    "failure_reason": _failure_reason(
                        verification,
                        extraction_error=extraction_error,
                        policy=policy,
                    ),
                    "error_log_path": error_log_value,
                    "generation_metadata": _to_jsonable(generation.metadata),
                    "generation_stage": "performance_search",
                    "search_round": search_round,
                    "parent_candidate_path": parent_attempt.get("candidate_path"),
                    "parent_candidate_id": parent_attempt.get("candidate_id"),
                    "parent_speedup_vs_eager": parent_speedup,
                    "improved_over_parent": improved,
                    "target_reached": _target_reached_from_summary(child_summary, config),
                }
                attempts.append(attempt)
                candidate_record = _candidate_json_record(
                    task=task,
                    config=config,
                    attempt=attempt,
                    backend=config.agent.backend,
                    model=config.agent.model,
                    selected_best=False,
                )
                candidate_records.append(candidate_record)
                if loaded is not None:
                    unload_candidate(loaded)
                if verification.passed and _attempt_quality_key(attempt) > _attempt_quality_key(best_attempt):
                    best_attempt = attempt
                    best_source = candidate_source
                global_candidate_index += 1
            if _target_reached(best_attempt, config):
                break
    return global_candidate_index


def _performance_search_parents(
    attempts: list[dict[str, Any]],
    initial_best_attempt: dict[str, Any],
    config: RunConfig,
) -> list[dict[str, Any]]:
    search_config = config.agent.performance_search
    if search_config.optimize_only_selected_best:
        return [initial_best_attempt]
    candidates = [
        attempt
        for attempt in attempts
        if attempt.get("verification", {}).get("passed") and _needs_performance_search(attempt, config)
    ]
    candidates.sort(key=_attempt_quality_key, reverse=True)
    return candidates[: max(search_config.keep_top_k, 1)]


def _template_copy_settings(config: RunConfig) -> list[dict[str, Any]]:
    grid = config.agent.performance_search.parameter_grid or {}
    block_sizes = grid.get("block_sizes") or [None]
    num_warps = grid.get("num_warps") or [None]
    contiguous_policies = grid.get("contiguous_policies") or ["preserve_template"]
    settings: list[dict[str, Any]] = []
    for block_size in block_sizes:
        for warp_count in num_warps:
            for policy in contiguous_policies:
                settings.append(
                    {
                        "block_size": block_size,
                        "num_warps": warp_count,
                        "contiguous_policy": policy,
                    }
                )
    limit = config.agent.performance_search.max_settings_per_task
    if limit is not None:
        settings = settings[: max(int(limit), 0)]
    return settings


def _load_and_verify(
    task: KernelTask,
    candidate_path: Path,
    candidate_name: str,
    config: RunConfig,
    verification_device: torch.device,
) -> tuple[LoadedCandidate | None, VerificationResult, list[str]]:
    error_chunks: list[str] = []
    try:
        loaded = load_candidate_from_path(candidate_path)
        if loaded.forward is None:
            raise CandidateLoadError(f"Candidate has no module-level forward: {candidate_path}")
        verification_shapes = _limited_shapes(task.benchmark_shapes, config.verification.max_shapes_per_task)
        verification = verify_candidate(
            task,
            loaded.forward,
            candidate_name=candidate_name,
            seeds=config.verification.seeds,
            shapes=verification_shapes,
            dtype=config.verification.dtype,
            device=verification_device,
        )
        if verification.error:
            error_chunks.append("Verification error:\n" + verification.error)
        elif not verification.passed:
            error_chunks.append("Verification failed:\n" + _format_verification_failures(verification))
        return loaded, verification, error_chunks
    except CandidateLoadError as exc:
        error_text = str(exc)
        error_chunks.append(error_text)
        return (
            None,
            VerificationResult(
                task_id=task.task_id,
                candidate_name=candidate_name,
                passed=False,
                error=error_text,
            ),
            error_chunks,
        )
    except Exception:
        error_text = traceback.format_exc()
        error_chunks.append(error_text)
        return (
            None,
            VerificationResult(
                task_id=task.task_id,
                candidate_name=candidate_name,
                passed=False,
                error=error_text,
            ),
            error_chunks,
        )


def _benchmark_candidate(
    task: KernelTask,
    loaded: LoadedCandidate,
    candidate_name: str,
    config: RunConfig,
    benchmark_device: torch.device,
) -> tuple[list[Any], list[str]]:
    benchmarks: list[Any] = []
    error_chunks: list[str] = []
    benchmark_shapes = _limited_shapes(task.benchmark_shapes, config.benchmark.max_shapes_per_task)
    if loaded.forward is None:
        raise CandidateLoadError(f"Candidate has no module-level forward: {loaded.path}")
    for shape in benchmark_shapes:
        benchmark = benchmark_task(
            task,
            loaded.forward,
            candidate_name=candidate_name,
            shape=shape,
            dtype=config.benchmark.dtype,
            device=benchmark_device,
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
        benchmarks.append(benchmark)
        if benchmark.benchmark_error:
            error_chunks.append("Benchmark error:\n" + benchmark.benchmark_error)
        if benchmark.compile_error:
            error_chunks.append("torch.compile error:\n" + benchmark.compile_error)
    return benchmarks, error_chunks


def _candidate_json_record(
    *,
    task: KernelTask,
    config: RunConfig,
    attempt: dict[str, Any],
    backend: str,
    model: str | None,
    selected_best: bool,
) -> dict[str, Any]:
    """Build the candidate-level JSONL record required for reproducible runs."""

    verification = attempt.get("verification") or {}
    benchmarks = attempt.get("benchmarks") or []
    policy = attempt.get("policy") or {}
    generation_metadata = attempt.get("generation_metadata") or {}
    return {
        "record_type": "candidate",
        "task_id": task.task_id,
        "task_name": task.name,
        "attempt_index": attempt.get("attempt_index"),
        "candidate_index": attempt.get("candidate_index"),
        "candidate_id": attempt.get("candidate_id"),
        "candidate_name": attempt.get("candidate_name"),
        "backend": backend,
        "model": model,
        "prompt_path": attempt.get("prompt_path"),
        "response_path": attempt.get("response_path"),
        "candidate_path": attempt.get("candidate_path"),
        "policy_passed": bool(policy.get("passed")),
        "policy_warnings": policy.get("warnings") or [],
        "policy_rejection_reason": policy.get("rejection_reason"),
        "verification_passed": bool(verification.get("passed")),
        "verification_summary": _verification_summary(verification),
        "benchmark_summary": _benchmark_summary(benchmarks),
        "selected_best": selected_best,
        "failure_reason": attempt.get("failure_reason"),
        "created_at": _now_iso(),
        "agent_type": config.agent.type,
        "prompt_version": config.agent.prompt_version,
        "repair_prompt_version": config.agent.repair_prompt_version,
        "performance_prompt_version": config.agent.performance_prompt_version,
        "generation_stage": attempt.get("generation_stage", "initial"),
        "search_round": attempt.get("search_round"),
        "parent_candidate_path": attempt.get("parent_candidate_path"),
        "parent_candidate_id": attempt.get("parent_candidate_id"),
        "parent_speedup_vs_eager": attempt.get("parent_speedup_vs_eager"),
        "improved_over_parent": attempt.get("improved_over_parent"),
        "best_initial_speedup_vs_eager": attempt.get("best_initial_speedup_vs_eager"),
        "best_final_speedup_vs_eager": attempt.get("best_final_speedup_vs_eager"),
        "performance_search_improvement_delta": attempt.get("performance_search_improvement_delta"),
        "target_reached": attempt.get("target_reached"),
        "template_family": attempt.get("template_family") or generation_metadata.get("template_family"),
        "task_family": attempt.get("task_family") or generation_metadata.get("task_family"),
        "block_size": attempt.get("block_size") or generation_metadata.get("block_size"),
        "reduction_block_size": (
            attempt.get("reduction_block_size")
            or generation_metadata.get("reduction_block_size")
        ),
        "reduction_axis": attempt.get("reduction_axis") or generation_metadata.get("reduction_axis"),
        "num_warps": attempt.get("num_warps") or generation_metadata.get("num_warps"),
        "contiguous_policy": (
            attempt.get("contiguous_policy") or generation_metadata.get("contiguous_policy")
        ),
        "template_id": attempt.get("template_id") or generation_metadata.get("template_id"),
        "num_stages": attempt.get("num_stages") or generation_metadata.get("num_stages"),
        "output_allocation_policy": (
            attempt.get("output_allocation_policy")
            or generation_metadata.get("output_allocation_policy")
        ),
        "shape_specialized": (
            attempt.get("shape_specialized")
            if attempt.get("shape_specialized") is not None
            else generation_metadata.get("shape_specialized")
        ),
        "feature_dim_mode": attempt.get("feature_dim_mode") or generation_metadata.get("feature_dim_mode"),
        "n_elements_mode": attempt.get("n_elements_mode") or generation_metadata.get("n_elements_mode"),
        "total_possible_variants": (
            attempt.get("total_possible_variants")
            or generation_metadata.get("total_possible_variants")
        ),
        "valid_possible_variants": (
            attempt.get("valid_possible_variants")
            or generation_metadata.get("valid_possible_variants")
        ),
        "generated_valid_variants": (
            attempt.get("generated_valid_variants")
            or generation_metadata.get("generated_valid_variants")
        ),
        "skipped_invalid_variants": (
            attempt.get("skipped_invalid_variants")
            or generation_metadata.get("skipped_invalid_variants")
        ),
        "skipped_reasons": attempt.get("skipped_reasons") or generation_metadata.get("skipped_reasons"),
        "variant_validation": (
            attempt.get("variant_validation")
            or generation_metadata.get("variant_validation")
        ),
        "actually_generated_variants": (
            attempt.get("actually_generated_variants")
            or generation_metadata.get("actually_generated_variants")
        ),
        "grid_sampling": attempt.get("grid_sampling") or generation_metadata.get("grid_sampling"),
        "sort_order": attempt.get("sort_order") or generation_metadata.get("sort_order"),
        "grid_was_capped": (
            attempt.get("grid_was_capped")
            if attempt.get("grid_was_capped") is not None
            else generation_metadata.get("grid_was_capped")
        ),
        "focused_seed_run": attempt.get("focused_seed_run") or generation_metadata.get("focused_seed_run"),
        "focused_seed_candidate_path": (
            attempt.get("focused_seed_candidate_path")
            or generation_metadata.get("focused_seed_candidate_path")
        ),
        "focused_rank": attempt.get("focused_rank") or generation_metadata.get("focused_rank"),
        "template_source_path": attempt.get("template_source_path"),
        "copied_from_template_id": attempt.get("copied_from_template_id"),
        "requested_block_size": attempt.get("requested_block_size"),
        "requested_num_warps": attempt.get("requested_num_warps"),
        "requested_contiguous_policy": attempt.get("requested_contiguous_policy"),
        "preserved_template_structure_score": attempt.get("preserved_template_structure_score"),
        "template_preservation": attempt.get("template_preservation"),
        "extra_torch_ops_detected": attempt.get("extra_torch_ops_detected"),
        "fallback_detected": attempt.get("fallback_detected"),
        "source_template_benchmark": attempt.get("source_template_benchmark"),
        "source_template_speedup_vs_eager": attempt.get("source_template_speedup_vs_eager"),
        "delta_vs_source_template": attempt.get("delta_vs_source_template"),
    }


def _policy_rejected_verification(
    task: KernelTask,
    candidate_name: str,
    policy: CandidatePolicyResult,
) -> VerificationResult:
    return VerificationResult(
        task_id=task.task_id,
        candidate_name=candidate_name,
        passed=False,
        error=f"policy_rejected:{policy.rejection_reason}",
    )


def _verification_summary(verification: dict[str, Any]) -> dict[str, Any]:
    cases = verification.get("cases") or []
    failed_cases = [case for case in cases if not case.get("passed")]
    max_abs = _max_optional(case.get("max_abs_error") for case in cases)
    max_rel = _max_optional(case.get("max_rel_error") for case in cases)
    first_failure = failed_cases[0] if failed_cases else {}
    return {
        "passed": bool(verification.get("passed")),
        "num_cases": len(cases),
        "num_failed_cases": len(failed_cases),
        "first_error_type": first_failure.get("error_type"),
        "first_message": first_failure.get("message") or verification.get("error"),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "elapsed_s": verification.get("elapsed_s"),
    }


def _benchmark_summary(benchmarks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not benchmarks:
        return None
    benchmark = benchmarks[0]
    eager = benchmark.get("eager") or {}
    candidate = benchmark.get("candidate") or {}
    compiled = benchmark.get("torch_compile") or {}
    speedup_compile = benchmark.get("speedup_vs_torch_compile")
    return {
        "shape": benchmark.get("shape"),
        "dtype": benchmark.get("dtype"),
        "device": benchmark.get("device"),
        "eager_median_ms": eager.get("median_ms"),
        "candidate_median_ms": candidate.get("median_ms"),
        "torch_compile_median_ms": compiled.get("median_ms"),
        "torch_compile_mode": benchmark.get("torch_compile_mode"),
        "timing_mode": benchmark.get("timing_mode"),
        "warmup": benchmark.get("warmup"),
        "repeat": benchmark.get("repeats"),
        "independent_sessions": benchmark.get("independent_sessions"),
        "cache_flush_enabled": benchmark.get("cache_flush_enabled"),
        "cache_flush_performed": benchmark.get("cache_flush_performed"),
        "candidate_ms_summary": benchmark.get("candidate_ms_summary"),
        "eager_ms_summary": benchmark.get("eager_ms_summary"),
        "torch_compile_ms_summary": benchmark.get("torch_compile_ms_summary"),
        "speedup_vs_eager": benchmark.get("speedup_vs_eager"),
        "speedup_vs_torch_compile": speedup_compile,
        "speedup_vs_compile": speedup_compile,
        "compile_time_ms": benchmark.get("compile_time_ms"),
        "runtime_only_ms": benchmark.get("runtime_only_ms"),
        "measurement_warnings": benchmark.get("measurement_warnings") or [],
        "session_summaries": benchmark.get("session_summaries") or [],
        "across_session_median_speedup": benchmark.get("across_session_median_speedup"),
        "across_session_iqr": benchmark.get("across_session_iqr"),
        "stable_above_eager": benchmark.get("stable_above_eager"),
        "stable_above_compile": benchmark.get("stable_above_compile"),
        "benchmark_error": benchmark.get("benchmark_error"),
        "compile_error": benchmark.get("compile_error"),
    }


def _select_best_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    if not attempts:
        raise RuntimeError("No attempts were recorded")
    correct = [attempt for attempt in attempts if attempt.get("verification", {}).get("passed")]
    if not correct:
        return attempts[-1]
    return max(correct, key=_attempt_quality_key)


def _attempt_quality_key(attempt: dict[str, Any]) -> tuple[float, float, float, float]:
    summary = _benchmark_summary(attempt.get("benchmarks") or []) or {}
    speedup_compile = summary.get("speedup_vs_torch_compile")
    speedup_eager = summary.get("speedup_vs_eager")
    candidate_median = summary.get("candidate_median_ms")
    has_compile = 1.0 if speedup_compile is not None else 0.0
    return (
        has_compile,
        float(speedup_compile) if speedup_compile is not None else float("-inf"),
        float(speedup_eager) if speedup_eager is not None else float("-inf"),
        -float(candidate_median) if candidate_median is not None else float("-inf"),
    )


def _mark_selected_best(
    candidate_records: list[dict[str, Any]],
    final_attempt: dict[str, Any],
) -> None:
    selected_id = final_attempt.get("candidate_id")
    selected_is_correct = bool(final_attempt.get("verification", {}).get("passed"))
    for record in candidate_records:
        record["selected_best"] = selected_is_correct and record.get("candidate_id") == selected_id


def _select_repair_entry(
    entries: list[dict[str, Any]],
    *,
    allow_slow_correct_repair: bool = True,
) -> dict[str, Any] | None:
    failed = [
        entry
        for entry in entries
        if not entry["attempt"].get("verification", {}).get("passed")
    ]
    if not failed:
        if not allow_slow_correct_repair:
            return None
        slow_correct = []
        for entry in entries:
            speedup = _attempt_speedup_vs_eager(entry["attempt"])
            if speedup is not None and speedup < 1.0:
                slow_correct.append(entry)
        if slow_correct:
            return min(slow_correct, key=lambda entry: _attempt_speedup_vs_eager(entry["attempt"]) or 0.0)
        return None
    return min(failed, key=_repair_priority)


def _repair_priority(entry: dict[str, Any]) -> tuple[int, int]:
    reason = entry["attempt"].get("failure_reason")
    priority = {
        "values_not_close": 0,
        "wrong_shape": 1,
        "wrong_dtype": 2,
        "nonfinite_output": 3,
        "bad_output_type": 4,
        "exception": 5,
        "code_extraction_failed": 6,
    }.get(str(reason), 10)
    has_cases = bool(entry["attempt"].get("verification", {}).get("cases"))
    return (priority, 0 if has_cases else 1)


def _attempt_speedup_vs_eager(attempt: dict[str, Any]) -> float | None:
    summary = _benchmark_summary(attempt.get("benchmarks") or []) or {}
    speedup = summary.get("speedup_vs_eager")
    return float(speedup) if speedup is not None else None


def _performance_feedback(attempt: dict[str, Any]) -> str | None:
    summary = _benchmark_summary(attempt.get("benchmarks") or []) or {}
    speedup = summary.get("speedup_vs_eager")
    if speedup is None:
        return None
    lines = [
        "Performance feedback:",
        "- Correctness passed but slow.",
        f"- speedup_vs_eager: {speedup}",
        f"- candidate_median_ms: {summary.get('candidate_median_ms')}",
        f"- eager_median_ms: {summary.get('eager_median_ms')}",
        f"- speedup_vs_torch_compile: {summary.get('speedup_vs_torch_compile')}",
    ]
    return "\n".join(lines)


def _needs_performance_search(attempt: dict[str, Any], config: RunConfig) -> bool:
    if not attempt.get("verification", {}).get("passed"):
        return False
    summary = _benchmark_summary(attempt.get("benchmarks") or []) or {}
    if not summary:
        return False
    return not _target_reached_from_summary(summary, config)


def _target_reached(attempt: dict[str, Any], config: RunConfig) -> bool:
    summary = _benchmark_summary(attempt.get("benchmarks") or []) or {}
    return _target_reached_from_summary(summary, config)


def _target_reached_from_summary(summary: dict[str, Any], config: RunConfig) -> bool:
    search_config = config.agent.performance_search
    eager_target = search_config.target_speedup_vs_eager
    compile_target = search_config.target_speedup_vs_compile
    eager_speedup = summary.get("speedup_vs_eager")
    compile_speedup = summary.get("speedup_vs_torch_compile")
    if eager_target is not None:
        if eager_speedup is None or float(eager_speedup) < float(eager_target):
            return False
    if compile_target is not None:
        if compile_speedup is None or float(compile_speedup) < float(compile_target):
            return False
    return eager_target is not None or compile_target is not None


def _speedup_greater(child_speedup: Any, parent_speedup: Any) -> bool:
    if child_speedup is None or parent_speedup is None:
        return False
    return float(child_speedup) > float(parent_speedup)


def _static_performance_flags(task_id: str, source: str) -> list[str]:
    if not source:
        return []
    try:
        from openkernelforge.reports.gpu_debrief import static_performance_flags
    except Exception:
        return []
    return static_performance_flags(task_id, source)


def _best_template_context(task_id: str, config: RunConfig) -> dict[str, Any] | None:
    search_config = config.agent.performance_search
    if not search_config.include_best_template_context:
        return None
    if not search_config.template_run_dir or "<template_run>" in str(search_config.template_run_dir):
        raise RuntimeError(
            "performance_search.include_best_template_context=true requires a real "
            "performance_search.template_run_dir. Run template autotune first with "
            "`python scripts/run_gpu_baseline_3tasks.py --config configs/template_3tasks_gpu_autotune.yaml "
            "--out-name template_3tasks_gpu_autotune`, then set template_run_dir to that run."
        )
    template_run = Path(search_config.template_run_dir)
    results_path = template_run / "results.jsonl"
    if not results_path.exists():
        raise RuntimeError(
            f"Template run directory is missing results.jsonl: {template_run}. "
            "Run template autotune first or fix performance_search.template_run_dir."
        )
    try:
        records = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Template run results.jsonl is malformed: {results_path}") from exc
    candidates = [
        record
        for record in records
        if record.get("record_type") == "candidate"
        and record.get("task_id") == task_id
        and record.get("verification_passed")
        and (record.get("benchmark_summary") or {}).get("speedup_vs_eager") is not None
    ]
    if not candidates:
        raise RuntimeError(
            f"No verified benchmarked template candidates found for task {task_id} in {template_run}."
        )
    best = max(
        candidates,
        key=_record_speedup_vs_eager,
    )
    candidate_path = Path(str(best.get("candidate_path")))
    if not candidate_path.is_absolute():
        candidate_path = template_run / candidate_path
    if not candidate_path.exists():
        candidate_path = Path(str(best.get("candidate_path")))
    return {
        "template_id": best.get("template_id"),
        "block_size": best.get("block_size"),
        "num_warps": best.get("num_warps"),
        "contiguous_policy": best.get("contiguous_policy"),
        "benchmark_summary": best.get("benchmark_summary"),
        "candidate_path": best.get("candidate_path"),
        "candidate_code": _read_text_path(candidate_path),
    }


def _read_text_path(path_value: Any) -> str:
    if not path_value:
        return ""
    try:
        path = Path(str(path_value))
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _annotate_performance_search_outcome(
    *,
    attempts: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    initial_best_attempt: dict[str, Any],
    final_attempt: dict[str, Any],
    config: RunConfig,
) -> None:
    initial_speedup = _attempt_speedup_vs_eager(initial_best_attempt)
    final_speedup = _attempt_speedup_vs_eager(final_attempt)
    delta = (
        final_speedup - initial_speedup
        if initial_speedup is not None and final_speedup is not None
        else None
    )
    target_reached = _target_reached(final_attempt, config)
    attempt_by_id = {attempt.get("candidate_id"): attempt for attempt in attempts}
    for attempt in attempts:
        attempt["best_initial_speedup_vs_eager"] = initial_speedup
        attempt["best_final_speedup_vs_eager"] = final_speedup
        attempt["performance_search_improvement_delta"] = delta
        attempt["target_reached"] = target_reached
    for record in candidate_records:
        attempt = attempt_by_id.get(record.get("candidate_id"), {})
        record["best_initial_speedup_vs_eager"] = initial_speedup
        record["best_final_speedup_vs_eager"] = final_speedup
        record["performance_search_improvement_delta"] = delta
        record["target_reached"] = target_reached
        for key in (
            "generation_stage",
            "search_round",
            "parent_candidate_path",
            "parent_candidate_id",
            "parent_speedup_vs_eager",
            "improved_over_parent",
        ):
            if key in attempt:
                record[key] = attempt.get(key)


def _max_optional(values: Any) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return max(numeric) if numeric else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _limited_shapes(shapes: list[Shape], limit: int) -> list[Shape]:
    if limit <= 0:
        raise ValueError("max_shapes_per_task must be positive")
    return shapes[:limit]


def _write_text_artifact(
    run_dir: Path,
    group: str,
    task_id: str,
    candidate_index: int,
    suffix: str,
    text: str,
) -> Path:
    if candidate_index < 0:
        raise ValueError("candidate_index must be non-negative")
    artifact_dir = resolve_task_artifact_dir(run_dir, group, task_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"candidate_{candidate_index:03d}{suffix}"
    path.write_text(text, encoding="utf-8")
    return path


def _write_error_log(
    run_dir: Path,
    task_id: str,
    candidate_index: int,
    error_chunks: list[str],
) -> str | None:
    if not error_chunks:
        return None
    log_dir = resolve_task_artifact_dir(run_dir, "logs", task_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = log_dir / f"candidate_{candidate_index:03d}.err.txt"
    error_log_path.write_text("\n\n".join(error_chunks), encoding="utf-8")
    return str(error_log_path)


def _select_final_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    if not attempts:
        raise RuntimeError("No attempts were recorded")
    for attempt in attempts:
        if attempt.get("verification", {}).get("passed"):
            return attempt
    return attempts[-1]


def _failure_reason(
    verification: VerificationResult,
    *,
    extraction_error: str | None = None,
    policy: CandidatePolicyResult | None = None,
) -> str | None:
    if policy is not None and not policy.passed:
        if policy.rejection_reason == "code_extraction_failed":
            return "code_extraction_failed"
        return "policy_rejected"
    if extraction_error:
        return "code_extraction_failed"
    if verification.passed:
        return None
    if verification.error:
        return "exception"
    for case in verification.cases:
        if not case.passed:
            return case.error_type or "verification_failed"
    return "verification_failed"


def _format_verification_failures(verification: VerificationResult) -> str:
    lines: list[str] = []
    for case in verification.cases:
        if case.passed:
            continue
        lines.append(
            "seed={seed} shape={shape} error_type={error_type} message={message} "
            "max_abs_error={max_abs_error} max_rel_error={max_rel_error}".format(
                seed=case.seed,
                shape=case.shape,
                error_type=case.error_type,
                message=case.message,
                max_abs_error=case.max_abs_error,
                max_rel_error=case.max_rel_error,
            )
        )
    return "\n".join(lines) if lines else "No case details were recorded."


def _make_run_dir(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        name = timestamp if suffix == 0 else f"{timestamp}_{suffix:02d}"
        run_dir = root / name
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique run directory under {root}")


def _write_run_metadata(
    run_dir: Path,
    *,
    config: RunConfig,
    environment: EnvironmentProbeResult,
    started_at: datetime,
    completed_at: datetime,
    records: list[dict[str, Any]],
) -> None:
    candidate_records = [
        candidate
        for record in records
        for candidate in record.get("candidate_records", [])
    ]
    skipped_by_task: dict[str, int] = {}
    for candidate in candidate_records:
        task_id = str(candidate.get("task_id"))
        if task_id not in skipped_by_task:
            skipped_by_task[task_id] = int(candidate.get("skipped_invalid_variants") or 0)
    metadata = {
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_s": (completed_at - started_at).total_seconds(),
        "task_count": len(records),
        "candidate_count": len(candidate_records),
        "skipped_invalid_variant_count": sum(skipped_by_task.values()),
        "policy_passed_count": sum(1 for record in candidate_records if record.get("policy_passed")),
        "verification_passed_count": sum(
            1 for record in candidate_records if record.get("verification_passed")
        ),
        "benchmarked_candidate_count": sum(
            1 for record in candidate_records if record.get("benchmark_summary")
        ),
        "performance_search_candidate_count": sum(
            1 for record in candidate_records if record.get("generation_stage") == "performance_search"
        ),
        "template_copy_candidate_count": sum(
            1 for record in candidate_records if record.get("generation_stage") == "template_copy"
        ),
        "performance_search_target_reached_tasks": len(
            {record.get("task_id") for record in candidate_records if record.get("target_reached")}
        ),
        "selected_correct_tasks": sum(
            1
            for record in records
            if record.get("verification", {}).get("passed")
        ),
        "environment_viability": environment.viability,
        "config": config.to_safe_dict(),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _write_environment_probe(run_dir: Path, environment: EnvironmentProbeResult) -> None:
    (run_dir / "environment_probe.json").write_text(
        json.dumps(environment.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )


def _enforce_execution_requirements(config: RunConfig, environment: EnvironmentProbeResult) -> None:
    failures: list[str] = []
    if config.execution.disabled_reason:
        failures.append(f"config disabled: {config.execution.disabled_reason}")
    if config.execution.require_cuda and not environment.cuda_available:
        failures.append("execution.require_cuda=true but CUDA is unavailable")
    if config.execution.require_triton and not environment.triton_available:
        failures.append("execution.require_triton=true but Triton is unavailable")
    if config.execution.require_tiny_triton_kernel and not environment.tiny_triton_kernel_passed:
        failures.append(
            "execution.require_tiny_triton_kernel=true but the tiny Triton kernel did not pass"
        )
    if failures:
        raise RuntimeError(
            "Execution environment requirements were not met: "
            + "; ".join(failures)
            + f". Environment viability: {environment.viability}"
        )


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    return value


def _record_speedup_vs_eager(record: dict[str, Any]) -> float:
    value = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
    if value is None:
        raise ValueError("Candidate record has no eager speedup")
    return float(value)
