"""Prompt construction helpers for future model-backed agents."""

from __future__ import annotations

from openkernelforge.tasks.base import KernelTask


PROMPT_V1_DEFAULT = "v1_default"
PROMPT_V2_TASK_SKELETONS = "v2_task_skeletons"


def build_task_prompt(
    task: KernelTask,
    *,
    allow_torch_fallback: bool = True,
    prompt_version: str = PROMPT_V1_DEFAULT,
) -> str:
    """Build a complete candidate-generation prompt from task metadata."""

    metadata = task.prompt_metadata()
    shapes = ", ".join(str(tuple(shape)) for shape in metadata["benchmark_shapes"])
    allowed_dtypes = ", ".join(metadata["allowed_dtypes"])
    source = metadata.get("reference_source") or "Reference source unavailable."
    task_hint = _task_hint(task.task_id)
    metadata_hints = _metadata_hints(metadata.get("metadata") or {})
    skeleton_hint = _task_skeleton_hint(task.task_id) if prompt_version == PROMPT_V2_TASK_SKELETONS else ""
    fallback_instruction = (
        "Torch fallback mode is enabled: a clear torch implementation is acceptable, "
        "but it must compute the actual output from inputs."
        if allow_torch_fallback
        else (
            "Torch fallback mode is disabled: do not use plain PyTorch fallback code "
            "such as direct tensor arithmetic or torch.relu/torch.add as the main implementation. "
            "Use Triton when CUDA is available. Torch may still be used for allocation, wrappers, "
            "shape handling, and CUDA tensors, for example torch.empty_like, torch.empty, "
            "and torch.empty_strided."
        )
    )
    return (
        "Write a complete Python candidate file for OpenKernelForge.\n"
        f"Prompt version: {prompt_version}\n"
        "The file must expose def forward(*args): ... at module scope.\n"
        "Allowed imports: torch, triton, and triton.language as tl.\n"
        "Do not fake outputs, cache expected tensors, read hidden files, or call the reference function.\n"
        "Prefer a simple correct Triton kernel over a complex broken kernel.\n"
        "For elementwise tasks, use a straightforward block-based Triton kernel.\n"
        "Include all required imports.\n"
        f"{fallback_instruction}\n"
        "Return only Python code, with no explanation outside code.\n\n"
        f"Task id: {metadata['task_id']}\n"
        f"Name: {metadata['name']}\n"
        f"Description: {metadata['description']}\n"
        f"Allowed dtypes: {allowed_dtypes}\n"
        f"Correctness tolerance: rtol={metadata['rtol']}, atol={metadata['atol']}\n"
        f"Benchmark shapes: {shapes}\n\n"
        f"Task-specific hint: {task_hint}\n"
        f"{metadata_hints}\n"
        f"{skeleton_hint}"
        "Reference implementation:\n"
        f"{source}\n"
    )


def _task_hint(task_id: str) -> str:
    hints = {
        "vector_add": "one output element per input element; compute out[i] = x[i] + y[i].",
        "relu": "elementwise max(x, 0); one output element per input element.",
        "bias_relu": "broadcast bias over the last dimension, add it to x, then apply relu.",
        "sigmoid_mul": "fuse sigmoid and multiply in one pass over x and z.",
        "add_relu": "fuse elementwise add and ReLU in one pass.",
        "residual_add_relu": "fuse x + residual + last-dimension bias, then ReLU.",
        "bias_gelu": "fuse last-dimension bias add and sigmoid-approximate GELU.",
        "row_sum": "reduce each row over the final dimension.",
        "layernorm_small": "one row per program; normalize over the final dimension and apply weight/bias.",
        "rmsnorm_small": "one row per program; compute RMS over the final dimension and apply weight.",
    }
    return hints.get(task_id, "match the reference semantics for every generated input shape.")


def _metadata_hints(metadata: dict) -> str:
    hints = metadata.get("prompt_hints") or []
    shape_metadata = metadata.get("shape_metadata") or {}
    lines = []
    if metadata.get("task_family"):
        lines.append(f"Task family: {metadata.get('task_family')}")
    if shape_metadata:
        lines.append(f"Shape metadata: {shape_metadata}")
    if hints:
        lines.append("Additional task hints:")
        lines.extend(f"- {hint}" for hint in hints)
    return "\n".join(lines) + "\n" if lines else ""


def _task_skeleton_hint(task_id: str) -> str:
    hints = {
        "vector_add": (
            "Task-specific Triton skeleton hint:\n"
            "- Flatten all tensors with x_flat = x.reshape(-1), y_flat = y.reshape(-1), and out_flat = out.reshape(-1).\n"
            "- Allocate output with output = torch.empty_like(x).\n"
            "- Use BLOCK_SIZE as a tl.constexpr meta-parameter.\n"
            "- Compute offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).\n"
            "- Use mask = offsets < n_elements.\n"
            "- Load x and y blocks, then store x + y to the output block.\n"
            "- Launch with grid = (triton.cdiv(n_elements, BLOCK_SIZE),).\n\n"
        ),
        "relu": (
            "Task-specific Triton skeleton hint:\n"
            "- Flatten input and output views so the kernel sees one contiguous logical vector.\n"
            "- Allocate output with output = torch.empty_like(x).\n"
            "- Use block vectorization: one program handles BLOCK_SIZE elements.\n"
            "- Load an x block, compute y = tl.maximum(x, 0.0), and store y.\n"
            "- Use offsets, mask, and grid = (triton.cdiv(n_elements, BLOCK_SIZE),).\n\n"
        ),
        "bias_relu": (
            "Task-specific Triton skeleton hint:\n"
            "- Input is likely shaped [rows, features] or similar; bias is indexed by the last dimension.\n"
            "- Flatten x and output to a logical vector.\n"
            "- Pass n_elements and features into forward/kernel; features = x.shape[-1].\n"
            "- Compute feature_idx = offsets % features to load bias[feature_idx].\n"
            "- Load x and bias, compute y = tl.maximum(x + bias, 0.0), then store y.\n"
            "- Use BLOCK_SIZE as tl.constexpr, offsets, mask, and a ceil-div grid.\n\n"
        ),
    }
    return hints.get(task_id, "")
