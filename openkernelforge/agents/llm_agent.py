"""LLM-backed candidate generation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openkernelforge.agents.backends import ModelBackend
from openkernelforge.agents.code_extract import CodeExtractionResult, extract_python_code
from openkernelforge.agents.performance import build_performance_prompt
from openkernelforge.agents.prompt_templates import build_task_prompt
from openkernelforge.agents.repair import build_repair_prompt
from openkernelforge.tasks.base import KernelTask

if TYPE_CHECKING:
    from openkernelforge.harness.verifier import VerificationResult


SYSTEM_PROMPT = (
    "You generate concise Python candidate kernels for local verification. "
    "Return code that can be imported as a Python module."
)


@dataclass(frozen=True)
class LLMGeneration:
    """A single raw backend response plus extracted candidate code."""

    attempt_index: int
    prompt: str
    raw_response: str
    extraction: CodeExtractionResult
    candidate_name: str
    metadata: dict[str, object] = field(default_factory=dict)


class LLMAgent:
    """Small provider-agnostic agent for generation and verifier-guided repair."""

    def __init__(
        self,
        backend: ModelBackend,
        *,
        backend_name: str = "unknown",
        max_attempts: int = 3,
        allow_torch_fallback: bool = True,
        temperature: float | None = 0.2,
        candidates_per_attempt: int = 1,
        prompt_version: str = "v1_default",
        repair_prompt_version: str = "v1_default",
        performance_prompt_version: str = "v1_cuda_elementwise_perf",
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if candidates_per_attempt <= 0:
            raise ValueError("candidates_per_attempt must be positive")
        self.backend = backend
        self.backend_name = backend_name
        self.max_attempts = max_attempts
        self.allow_torch_fallback = allow_torch_fallback
        self.temperature = temperature
        self.candidates_per_attempt = candidates_per_attempt
        self.prompt_version = prompt_version
        self.repair_prompt_version = repair_prompt_version
        self.performance_prompt_version = performance_prompt_version

    def initial_prompt(self, task: KernelTask) -> str:
        return build_task_prompt(
            task,
            allow_torch_fallback=self.allow_torch_fallback,
            prompt_version=self.prompt_version,
        )

    def generate_initial(
        self,
        task: KernelTask,
        *,
        attempt_index: int = 0,
        candidate_index: int = 0,
    ) -> LLMGeneration:
        prompt = self.initial_prompt(task)
        return self._generate(
            task,
            prompt=prompt,
            attempt_index=attempt_index,
            candidate_index=candidate_index,
            stage="initial",
        )

    def generate_repair(
        self,
        task: KernelTask,
        *,
        attempt_index: int,
        candidate_index: int = 0,
        original_task_prompt: str,
        previous_candidate: str,
        verification: VerificationResult | None,
        extra_failure: str | None = None,
    ) -> LLMGeneration:
        prompt = build_repair_prompt(
            task=task,
            original_task_prompt=original_task_prompt,
            previous_candidate=previous_candidate,
            verification=verification,
            extra_failure=extra_failure,
            repair_prompt_version=self.repair_prompt_version,
        )
        return self._generate(
            task,
            prompt=prompt,
            attempt_index=attempt_index,
            candidate_index=candidate_index,
            stage="repair",
        )

    def generate_performance_candidate(
        self,
        task: KernelTask,
        *,
        attempt_index: int,
        candidate_index: int,
        previous_candidate: str,
        benchmark_summary: dict[str, object] | None,
        heuristic_flags: list[str] | None = None,
        template_context: dict[str, object] | None = None,
    ) -> LLMGeneration:
        prompt = build_performance_prompt(
            task=task,
            previous_candidate=previous_candidate,
            benchmark_summary=benchmark_summary,
            heuristic_flags=heuristic_flags,
            template_context=template_context,
            performance_prompt_version=self.performance_prompt_version,
        )
        return self._generate(
            task,
            prompt=prompt,
            attempt_index=attempt_index,
            candidate_index=candidate_index,
            stage="performance_search",
        )

    def generate_template_copy_candidate(
        self,
        task: KernelTask,
        *,
        attempt_index: int,
        candidate_index: int,
        template_code: str,
        template_benchmark_summary: dict[str, object] | None,
        requested_parameters: dict[str, object],
        heuristic_flags: list[str] | None = None,
        template_context: dict[str, object] | None = None,
    ) -> LLMGeneration:
        context = dict(template_context or {})
        context["candidate_code"] = template_code
        context["requested_parameters"] = dict(requested_parameters)
        prompt = build_performance_prompt(
            task=task,
            previous_candidate=template_code,
            benchmark_summary=template_benchmark_summary,
            heuristic_flags=heuristic_flags,
            template_context=context,
            performance_prompt_version=self.performance_prompt_version,
        )
        return self._generate(
            task,
            prompt=prompt,
            attempt_index=attempt_index,
            candidate_index=candidate_index,
            stage="template_copy",
        )

    def _generate(
        self,
        task: KernelTask,
        *,
        prompt: str,
        attempt_index: int,
        candidate_index: int,
        stage: str,
    ) -> LLMGeneration:
        raw_response = self.backend.generate(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=self.temperature,
        )
        extraction = extract_python_code(raw_response)
        return LLMGeneration(
            attempt_index=attempt_index,
            prompt=prompt,
            raw_response=raw_response,
            extraction=extraction,
            candidate_name=(
                f"llm_{self.backend_name}_{task.task_id}_"
                f"a{attempt_index:03d}_c{candidate_index:03d}"
            ),
            metadata={
                "agent": "llm",
                "backend": self.backend_name,
                "stage": stage,
                "attempt_index": attempt_index,
                "candidate_index": candidate_index,
                "temperature": self.temperature,
                "allow_torch_fallback": self.allow_torch_fallback,
                "prompt_version": self.prompt_version,
                "repair_prompt_version": self.repair_prompt_version,
                "performance_prompt_version": self.performance_prompt_version,
            },
        )
