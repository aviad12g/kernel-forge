"""Repair prompt helpers for verifier-guided candidate revision."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openkernelforge.tasks.base import KernelTask

if TYPE_CHECKING:
    from openkernelforge.harness.verifier import VerificationResult


REPAIR_V1_DEFAULT = "v1_default"
REPAIR_V3_CUDA = "v3_cuda_repair"


def summarize_verification_failure(verification: VerificationResult | None) -> str:
    """Convert structured verifier output into compact repair feedback."""

    if verification is None:
        return "No verifier result is available."

    lines = [
        f"Verification passed: {verification.passed}",
        f"Task id: {verification.task_id}",
        f"Candidate name: {verification.candidate_name}",
    ]
    if verification.error:
        lines.append("Exception or top-level error:")
        lines.append(verification.error)

    failed_cases = [case for case in verification.cases if not case.passed]
    if not failed_cases and not verification.error:
        lines.append("No failed cases were recorded.")

    for index, case in enumerate(failed_cases, start=1):
        lines.append(f"Failed case {index}:")
        lines.append(f"- seed: {case.seed}")
        lines.append(f"- shape: {case.shape}")
        lines.append(f"- error_type: {case.error_type}")
        lines.append(f"- message: {case.message}")
        lines.append(f"- output_shape: {case.output_shape}")
        lines.append(f"- reference_shape: {case.reference_shape}")
        lines.append(f"- output_dtype: {case.output_dtype}")
        lines.append(f"- reference_dtype: {case.reference_dtype}")
        lines.append(f"- max_abs_error: {case.max_abs_error}")
        lines.append(f"- max_rel_error: {case.max_rel_error}")
    return "\n".join(lines)


def build_repair_prompt(
    *,
    task: KernelTask,
    original_task_prompt: str,
    previous_candidate: str,
    verification: VerificationResult | None,
    extra_failure: str | None = None,
    repair_prompt_version: str = REPAIR_V1_DEFAULT,
) -> str:
    """Build a repair prompt from task context, code, and verifier feedback."""

    feedback = summarize_verification_failure(verification)
    extra = f"\nAdditional failure detail:\n{extra_failure}\n" if extra_failure else ""
    cuda_guidance = _cuda_repair_guidance(verification, extra_failure) if repair_prompt_version == REPAIR_V3_CUDA else ""
    return (
        "You are repairing a Python candidate kernel for OpenKernelForge.\n"
        f"Repair prompt version: {repair_prompt_version}\n"
        "Return only corrected Python code. Do not include Markdown fences or explanation.\n"
        "The corrected file must expose def forward(*args): ... and may import only "
        "torch, triton, and triton.language as tl.\n\n"
        "Original task prompt:\n"
        f"{original_task_prompt}\n\n"
        "Previous candidate code:\n"
        "```python\n"
        f"{previous_candidate.rstrip()}\n"
        "```\n\n"
        "Verifier feedback:\n"
        f"{feedback}\n"
        f"{extra}\n"
        f"{cuda_guidance}"
        "Fix the candidate so it matches the reference for all checked shapes, dtypes, and seeds.\n"
    )


def _cuda_repair_guidance(
    verification: VerificationResult | None,
    extra_failure: str | None,
) -> str:
    text = (extra_failure or "").lower()
    lines = ["CUDA/Triton repair guidance:"]
    if verification is not None and verification.passed:
        lines.extend(
            [
                "- Correctness passed but performance is slower than the baseline.",
                "- Use the benchmark feedback above, especially speedup vs eager and median runtime.",
                "- Produce a simpler/faster Triton implementation.",
                "- Reduce Python wrapper overhead and avoid unnecessary torch operations in forward except output allocation.",
                "- Use contiguous flattening for simple elementwise tasks.",
                "- Try larger BLOCK_SIZE values such as 256, 512, or 1024 where appropriate.",
            ]
        )
    if "triton" in text and ("compile" in text or "compilation" in text or "traceback" in text):
        lines.extend(
            [
                "- Triton compile traceback guidance: inspect the traceback and fix unsupported Triton JIT usage.",
                "- Use @triton.jit for kernels and tl.constexpr for compile-time constants.",
                "- Avoid unsupported Python, dynamic Python control flow, or non-Triton functions inside JIT kernels.",
                "- Keep kernel arguments simple tensors/scalars and move wrapper logic outside the JIT function.",
            ]
        )
    if len(lines) == 1:
        lines.append("- Return only Python code with a top-level forward(*args).")
    lines.append("- Return only Python code.")
    return "\n" + "\n".join(lines) + "\n"
