"""Failure taxonomy for OpenKernelForge candidate records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EXTRACTION_FAILURE = "EXTRACTION_FAILURE"
POLICY_REJECTED_TORCH_FALLBACK = "POLICY_REJECTED_TORCH_FALLBACK"
POLICY_REJECTED_OTHER = "POLICY_REJECTED_OTHER"
SYNTAX_ERROR = "SYNTAX_ERROR"
IMPORT_ERROR = "IMPORT_ERROR"
ENV_MISSING_TRITON = "ENV_MISSING_TRITON"
ENV_MISSING_CUDA = "ENV_MISSING_CUDA"
ENV_TRITON_EXECUTION_FAILED = "ENV_TRITON_EXECUTION_FAILED"
MODEL_IMPORT_ERROR = "MODEL_IMPORT_ERROR"
TRITON_COMPILE_ERROR = "TRITON_COMPILE_ERROR"
RUNTIME_ERROR = "RUNTIME_ERROR"
SHAPE_MISMATCH = "SHAPE_MISMATCH"
DTYPE_MISMATCH = "DTYPE_MISMATCH"
NUMERICAL_MISMATCH = "NUMERICAL_MISMATCH"
NAN_OR_INF = "NAN_OR_INF"
TIMEOUT = "TIMEOUT"
BENCHMARK_ERROR = "BENCHMARK_ERROR"
CORRECT_BUT_SLOW = "CORRECT_BUT_SLOW"
CORRECT_PROMISING_BUT_SLOW = "CORRECT_PROMISING_BUT_SLOW"
CORRECT_AND_FAST = "CORRECT_AND_FAST"
UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


@dataclass(frozen=True)
class FailureClassification:
    failure_type: str
    short_reason: str
    evidence: list[str] = field(default_factory=list)
    is_training_useful: bool = False
    suggested_dataset_split: str = "analysis_only"


def classify_candidate_record(
    record: dict[str, Any],
    *,
    correct_fast_threshold_vs_eager: float = 1.0,
    correct_promising_threshold_vs_eager: float = 0.8,
) -> FailureClassification:
    """Classify a candidate-level record into a stable research taxonomy."""

    evidence: list[str] = []
    failure_reason = str(record.get("failure_reason") or "")
    policy_reason = str(record.get("policy_rejection_reason") or "")
    verification = record.get("verification_summary") or {}
    benchmark = record.get("benchmark_summary") or {}
    first_error = str(verification.get("first_error_type") or "")
    first_message = str(verification.get("first_message") or "")
    log_text = _read_log(record.get("error_log_path"))
    candidate_code = _read_log(record.get("candidate_path"))
    environment = record.get("environment_probe") or {}
    haystack = " ".join(
        [failure_reason, policy_reason, first_error, first_message, log_text, candidate_code]
    )

    if policy_reason:
        evidence.append(f"policy_rejection_reason={policy_reason}")
        if "torch_fallback" in policy_reason or "direct_" in policy_reason:
            return FailureClassification(
                failure_type=POLICY_REJECTED_TORCH_FALLBACK,
                short_reason="Policy rejected obvious torch fallback.",
                evidence=evidence,
                is_training_useful=True,
                suggested_dataset_split="rejected",
            )
        return FailureClassification(
            failure_type=POLICY_REJECTED_OTHER,
            short_reason="Policy rejected candidate.",
            evidence=evidence,
            is_training_useful=True,
            suggested_dataset_split="rejected",
        )

    if failure_reason == "code_extraction_failed" or not record.get("policy_passed", True):
        evidence.append(f"failure_reason={failure_reason}")
        return FailureClassification(
            failure_type=EXTRACTION_FAILURE,
            short_reason="Candidate code could not be extracted.",
            evidence=evidence,
            is_training_useful=True,
            suggested_dataset_split="rejected",
        )

    if benchmark.get("benchmark_error") or benchmark.get("compile_error"):
        evidence.append("benchmark_error_or_compile_error")
        return FailureClassification(
            failure_type=BENCHMARK_ERROR,
            short_reason="Candidate failed during benchmarking.",
            evidence=evidence,
            is_training_useful=True,
            suggested_dataset_split="analysis_only",
        )

    if record.get("verification_passed"):
        speedup = benchmark.get("speedup_vs_eager")
        if speedup is not None and float(speedup) < correct_promising_threshold_vs_eager:
            evidence.append(f"speedup_vs_eager={speedup}")
            return FailureClassification(
                failure_type=CORRECT_BUT_SLOW,
                short_reason="Candidate is correct but slower than PyTorch eager.",
                evidence=evidence,
                is_training_useful=True,
                suggested_dataset_split="optimization",
            )
        if speedup is not None and float(speedup) < correct_fast_threshold_vs_eager:
            evidence.append(f"speedup_vs_eager={speedup}")
            return FailureClassification(
                failure_type=CORRECT_PROMISING_BUT_SLOW,
                short_reason="Candidate is correct and close enough to be useful optimization data.",
                evidence=evidence,
                is_training_useful=True,
                suggested_dataset_split="optimization",
            )
        if speedup is not None:
            evidence.append(f"speedup_vs_eager={speedup}")
        if record.get("selected_best"):
            evidence.append("selected_best=true")
        return FailureClassification(
            failure_type=CORRECT_AND_FAST,
            short_reason="Candidate is correct and suitable for raw SFT review.",
            evidence=evidence,
            is_training_useful=True,
            suggested_dataset_split="sft_raw",
        )

    lowered = haystack.lower()
    if "syntaxerror" in lowered or "syntax error" in lowered:
        return _classified(SYNTAX_ERROR, "Candidate has Python syntax error.", haystack, "repair")
    if _looks_like_triton_candidate(candidate_code, lowered):
        if environment.get("triton_available") is False or "no module named 'triton'" in lowered:
            return _classified(
                ENV_MISSING_TRITON,
                "Candidate needs Triton, but Triton is not available in the local environment.",
                haystack,
                "analysis_only",
            )
        if environment.get("cuda_available") is False:
            return _classified(
                ENV_MISSING_CUDA,
                "Candidate needs CUDA/Triton execution, but CUDA is not available locally.",
                haystack,
                "analysis_only",
            )
        if (
            environment.get("triton_available") is True
            and environment.get("cuda_available") is True
            and environment.get("tiny_triton_kernel_passed") is False
        ):
            return _classified(
                ENV_TRITON_EXECUTION_FAILED,
                "Local Triton imports, but the environment cannot execute a tiny Triton kernel.",
                haystack,
                "analysis_only",
            )
    if "importerror" in lowered or "modulenotfounderror" in lowered:
        if environment.get("triton_available") is True:
            return _classified(
                MODEL_IMPORT_ERROR,
                "Candidate imported a module that is missing or invalid for this environment.",
                haystack,
                "repair",
            )
        return _classified(IMPORT_ERROR, "Candidate failed during import.", haystack, "repair")
    if "triton" in lowered and ("compile" in lowered or "compilation" in lowered):
        return _classified(TRITON_COMPILE_ERROR, "Triton compilation failed.", haystack, "repair")
    if "timeout" in lowered:
        return _classified(TIMEOUT, "Candidate timed out.", haystack, "analysis_only")
    if first_error == "wrong_shape":
        return _classified(SHAPE_MISMATCH, "Candidate returned wrong shape.", haystack, "repair")
    if first_error == "wrong_dtype":
        return _classified(DTYPE_MISMATCH, "Candidate returned wrong dtype.", haystack, "repair")
    if first_error == "values_not_close":
        return _classified(NUMERICAL_MISMATCH, "Candidate output differs numerically.", haystack, "repair")
    if first_error in {"nonfinite_output", "nonfinite_reference"}:
        return _classified(NAN_OR_INF, "Candidate produced NaN or Inf.", haystack, "repair")
    if first_error == "exception" or failure_reason == "exception":
        return _classified(RUNTIME_ERROR, "Candidate raised an exception.", haystack, "repair")

    return FailureClassification(
        failure_type=UNKNOWN_FAILURE,
        short_reason="Candidate failed for an unclassified reason.",
        evidence=[item for item in [failure_reason, first_error, first_message] if item],
        is_training_useful=False,
        suggested_dataset_split="analysis_only",
    )


def _classified(failure_type: str, reason: str, evidence_text: str, split: str) -> FailureClassification:
    return FailureClassification(
        failure_type=failure_type,
        short_reason=reason,
        evidence=[evidence_text[:500]] if evidence_text else [],
        is_training_useful=True,
        suggested_dataset_split=split,
    )


def _read_log(path_value: Any) -> str:
    if not path_value:
        return ""
    try:
        from pathlib import Path

        path = Path(str(path_value))
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return ""


def _looks_like_triton_candidate(candidate_code: str, lowered_haystack: str) -> bool:
    lowered_code = candidate_code.lower()
    return (
        "import triton" in lowered_code
        or "triton.language" in lowered_code
        or "@triton.jit" in lowered_code
        or "triton" in lowered_haystack
    )
