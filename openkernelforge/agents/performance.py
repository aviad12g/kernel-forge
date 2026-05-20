"""Performance-optimization prompts for correct-but-slow candidates."""

from __future__ import annotations

from typing import Any

from openkernelforge.tasks.base import KernelTask


PERFORMANCE_PROMPT_V1_CUDA_ELEMENTWISE = "v1_cuda_elementwise_perf"
PERFORMANCE_PROMPT_V3_STRICT_TEMPLATE_COPY = "v3_strict_template_copy"


def build_performance_prompt(
    *,
    task: KernelTask,
    previous_candidate: str,
    benchmark_summary: dict[str, Any] | None,
    heuristic_flags: list[str] | None = None,
    template_context: dict[str, Any] | None = None,
    performance_prompt_version: str = PERFORMANCE_PROMPT_V1_CUDA_ELEMENTWISE,
) -> str:
    """Build a prompt for optimizing a verified but slow candidate."""

    if performance_prompt_version == PERFORMANCE_PROMPT_V3_STRICT_TEMPLATE_COPY:
        return _strict_template_copy_prompt(
            task=task,
            previous_candidate=previous_candidate,
            benchmark_summary=benchmark_summary,
            heuristic_flags=heuristic_flags,
            template_context=template_context,
        )

    benchmark = benchmark_summary or {}
    flags = heuristic_flags or []
    flag_text = "\n".join(f"- {flag}" for flag in flags) if flags else "- none detected"
    template_text = _template_context_text(template_context)
    return (
        "You are optimizing a correct OpenKernelForge Triton candidate.\n"
        f"Performance prompt version: {performance_prompt_version}\n"
        "Return only Python code. Do not include Markdown fences or explanation.\n"
        "Keep the same forward(*args) API and preserve correctness exactly.\n"
        "The previous candidate passed correctness but is too slow.\n"
        "Write a faster Triton implementation, not a different high-level algorithm.\n"
        "Do not use plain PyTorch fallback. Do not call the reference implementation.\n"
        "Torch is allowed only for output allocation, wrappers, and shape handling.\n"
        "Avoid extra torch ops in forward except output allocation and shape handling.\n"
        "Use contiguous flattening for simple elementwise operations.\n"
        "Try larger BLOCK_SIZE values such as 256, 512, or 1024 where appropriate.\n"
        "Use one Triton program per block, not one program per element.\n"
        "Use tl.constexpr for BLOCK_SIZE and keep wrapper logic minimal.\n"
        "Do not add safety fallback branches.\n\n"
        f"Task id: {task.task_id}\n"
        f"Task name: {task.name}\n"
        f"Task description: {task.description}\n"
        f"Task-specific operation:\n{_task_perf_hint(task.task_id)}\n\n"
        "Benchmark feedback for the previous candidate:\n"
        f"- candidate_median_ms: {benchmark.get('candidate_median_ms')}\n"
        f"- eager_median_ms: {benchmark.get('eager_median_ms')}\n"
        f"- torch_compile_median_ms: {benchmark.get('torch_compile_median_ms')}\n"
        f"- speedup_vs_eager: {benchmark.get('speedup_vs_eager')}\n"
        f"- speedup_vs_torch_compile: {benchmark.get('speedup_vs_torch_compile')}\n\n"
        "Static heuristic flags from prior source inspection. These are hints, not profiler facts:\n"
        f"{flag_text}\n\n"
        f"{template_text}"
        "Previous correct-but-slow candidate code:\n"
        "```python\n"
        f"{previous_candidate.rstrip()}\n"
        "```\n"
    )


def _task_perf_hint(task_id: str) -> str:
    hints = {
        "vector_add": (
            "- Flatten x, y, and output.\n"
            "- Use output = torch.empty_like(x).\n"
            "- Use one load from x, one load from y, and one store.\n"
            "- Avoid extra intermediate allocations.\n"
            "- Build grid from total numel."
        ),
        "relu": (
            "- Flatten x and output.\n"
            "- Use tl.maximum(vals, 0.0).\n"
            "- Use one load and one store.\n"
            "- Do not use torch.relu or torch.maximum in forward."
        ),
        "bias_relu": (
            "- Flatten x and output.\n"
            "- Compute feature_idx = offsets % feature_dim.\n"
            "- Load bias[feature_idx].\n"
            "- Compute tl.maximum(x + bias, 0.0).\n"
            "- Avoid Python loops and avoid expanding bias with torch."
        ),
        "sigmoid_mul": (
            "- Flatten x, z, and output.\n"
            "- Compute sigmoid with Triton math, then multiply by z.\n"
            "- Use one load from x, one load from z, and one store."
        ),
        "add_relu": (
            "- Flatten x, y, and output.\n"
            "- Compute tl.maximum(x + y, 0.0).\n"
            "- Avoid torch.relu or direct torch arithmetic in forward."
        ),
        "residual_add_relu": (
            "- Flatten x, residual, and output.\n"
            "- Compute feature_idx = offsets % feature_dim for bias.\n"
            "- Compute tl.maximum(x + residual + bias, 0.0)."
        ),
        "bias_gelu": (
            "- Flatten x and output.\n"
            "- Compute feature_idx = offsets % feature_dim for bias.\n"
            "- Use the sigmoid GELU approximation matching the reference."
        ),
        "row_sum": (
            "- Use one Triton program per row.\n"
            "- Load the final dimension block and reduce with tl.sum.\n"
            "- Keep feature_dim as tl.constexpr where possible."
        ),
        "layernorm_small": (
            "- Use one Triton program per row.\n"
            "- Reduce mean and variance over the final dimension with tl.sum.\n"
            "- Load weight and bias once per row block."
        ),
        "rmsnorm_small": (
            "- Use one Triton program per row.\n"
            "- Reduce mean square over the final dimension and apply weight."
        ),
    }
    return hints.get(task_id, "- Preserve the reference semantics while reducing wrapper and kernel overhead.")


def _template_context_text(template_context: dict[str, Any] | None) -> str:
    if not template_context:
        return ""
    return (
        "Best deterministic template context for this same task:\n"
        "- This template is known-correct in a previous template autotune run.\n"
        "- Improve or adapt this known-correct template; do not add wrapper overhead.\n"
        f"- template_id: {template_context.get('template_id')}\n"
        f"- block_size: {template_context.get('block_size')}\n"
        f"- num_warps: {template_context.get('num_warps')}\n"
        f"- contiguous_policy: {template_context.get('contiguous_policy')}\n"
        f"- benchmark_summary: {template_context.get('benchmark_summary')}\n"
        "Best template source code:\n"
        "```python\n"
        f"{str(template_context.get('candidate_code') or '').rstrip()}\n"
        "```\n\n"
    )


def _strict_template_copy_prompt(
    *,
    task: KernelTask,
    previous_candidate: str,
    benchmark_summary: dict[str, Any] | None,
    heuristic_flags: list[str] | None,
    template_context: dict[str, Any] | None,
) -> str:
    benchmark = benchmark_summary or {}
    flags = heuristic_flags or []
    flag_text = "\n".join(f"- {flag}" for flag in flags) if flags else "- none detected"
    template = template_context or {}
    requested = template.get("requested_parameters") or {}
    template_code = str(template.get("candidate_code") or previous_candidate or "").rstrip()
    return (
        "You are copying/adapting a known-good OpenKernelForge Triton template.\n"
        f"Performance prompt version: {PERFORMANCE_PROMPT_V3_STRICT_TEMPLATE_COPY}\n"
        "Return only Python code. Do not include Markdown fences or explanation.\n\n"
        "Strict template-copy rules:\n"
        "- Preserve the template structure exactly unless a requested parameter change requires editing it.\n"
        "- Do not rewrite the wrapper.\n"
        "- Do not add try/except.\n"
        "- Do not add PyTorch fallback.\n"
        "- Do not add extra torch ops.\n"
        "- Do not add torch.relu, torch.maximum, torch.add, torch.matmul, or torch.sigmoid.\n"
        "- Do not add .contiguous() unless the template already uses it.\n"
        "- Keep exactly one Triton kernel launch.\n"
        "- Keep the same grid logic.\n"
        "- Keep the same flattening/indexing logic.\n"
        "- Keep the same BLOCK_SIZE and num_warps unless this prompt asks you to vary them.\n"
        "- Keep forward(*args) as the public API.\n"
        "- Do not call the reference implementation.\n"
        "- Preserve correctness exactly.\n\n"
        "Only allowed modifications:\n"
        "1. vary BLOCK_SIZE if requested\n"
        "2. vary num_warps if requested\n"
        "3. remove unnecessary wrapper overhead\n"
        "4. task-specific small changes that preserve correctness\n\n"
        f"Task id: {task.task_id}\n"
        f"Task name: {task.name}\n"
        f"Task description: {task.description}\n"
        f"Task-specific operation:\n{_task_perf_hint(task.task_id)}\n\n"
        "Requested parameter setting:\n"
        f"- requested_block_size: {requested.get('block_size')}\n"
        f"- requested_num_warps: {requested.get('num_warps')}\n"
        f"- requested_contiguous_policy: {requested.get('contiguous_policy')}\n\n"
        "Source template benchmark:\n"
        f"- candidate_median_ms: {benchmark.get('candidate_median_ms')}\n"
        f"- eager_median_ms: {benchmark.get('eager_median_ms')}\n"
        f"- torch_compile_median_ms: {benchmark.get('torch_compile_median_ms')}\n"
        f"- speedup_vs_eager: {benchmark.get('speedup_vs_eager')}\n"
        f"- speedup_vs_torch_compile: {benchmark.get('speedup_vs_torch_compile')}\n\n"
        "Static heuristic notes. These are source heuristics, not profiler facts:\n"
        f"{flag_text}\n\n"
        "Best known template source code to copy/adapt:\n"
        "```python\n"
        f"{template_code}\n"
        "```\n"
    )
