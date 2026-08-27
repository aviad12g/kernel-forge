"""Fused internal benchmark tasks for Triton performance experiments."""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F

from openkernelforge.tasks.base import KernelTask, Shape, TaskTolerance


FUSED_ELEMENTWISE_SHAPE: Shape = (4096, 1024)
FUSED_REDUCTION_SHAPE: Shape = (4096, 1024)
FUSED_NORM_SHAPE: Shape = (4096, 1024)


def _make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _randn(
    generator: torch.Generator,
    shape: Shape,
    dtype: torch.dtype,
    device: torch.device,
    scale: float = 1.0,
) -> torch.Tensor:
    tensor = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    return tensor.to(device=device, dtype=dtype)


def _source(fn) -> str | None:
    try:
        return inspect.getsource(fn)
    except OSError:
        return None


def bias_relu_ref(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.relu(x + bias)


def sigmoid_mul_ref(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x) * z


def add_relu_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x + y)


def residual_add_relu_ref(x: torch.Tensor, residual: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.relu(x + residual + bias)


def bias_gelu_ref(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    shifted = x + bias
    return shifted * torch.sigmoid(1.702 * shifted)


def row_sum_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x, dim=-1)


def layernorm_small_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=1e-5)


def rmsnorm_small_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)
    return x * rms * weight


def _two_tensor_inputs(
    seed: int,
    shape: Shape,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = _make_generator(seed)
    return _randn(generator, shape, dtype, device), _randn(generator, shape, dtype, device)


def _bias_inputs(
    seed: int,
    shape: Shape,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(shape) != 2:
        raise ValueError("bias fused tasks expect shape (rows, features)")
    generator = _make_generator(seed)
    rows, features = shape
    return _randn(generator, (rows, features), dtype, device), _randn(generator, (features,), dtype, device)


def residual_add_relu_inputs(
    seed: int,
    shape: Shape,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(shape) != 2:
        raise ValueError("residual_add_relu expects shape (rows, features)")
    generator = _make_generator(seed)
    rows, features = shape
    return (
        _randn(generator, (rows, features), dtype, device),
        _randn(generator, (rows, features), dtype, device),
        _randn(generator, (features,), dtype, device),
    )


def row_sum_inputs(seed: int, shape: Shape, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor]:
    if len(shape) != 2:
        raise ValueError("row_sum expects shape (rows, features)")
    generator = _make_generator(seed)
    return (_randn(generator, shape, dtype, device),)


def layernorm_small_inputs(
    seed: int,
    shape: Shape,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(shape) != 2:
        raise ValueError("layernorm_small expects shape (rows, features)")
    generator = _make_generator(seed)
    rows, features = shape
    return (
        _randn(generator, (rows, features), dtype, device),
        _randn(generator, (features,), dtype, device, scale=0.25) + 1.0,
        _randn(generator, (features,), dtype, device, scale=0.1),
    )


def rmsnorm_small_inputs(
    seed: int,
    shape: Shape,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(shape) != 2:
        raise ValueError("rmsnorm_small expects shape (rows, features)")
    generator = _make_generator(seed)
    rows, features = shape
    return (
        _randn(generator, (rows, features), dtype, device),
        _randn(generator, (features,), dtype, device, scale=0.25) + 1.0,
    )


def get_fused_tasks() -> list[KernelTask]:
    """Return the internal fused8 task set."""

    return [
        _task(
            "bias_relu",
            "Fused Bias ReLU",
            "Compute relu(x + bias) with bias broadcast across the last dimension.",
            bias_relu_ref,
            _bias_inputs,
            FUSED_ELEMENTWISE_SHAPE,
            "elementwise_fusion",
            "feature_dim",
            tolerance=TaskTolerance(rtol=1e-5, atol=1e-6),
        ),
        _task(
            "sigmoid_mul",
            "Fused Sigmoid Multiply",
            "Compute sigmoid(x) * z elementwise.",
            sigmoid_mul_ref,
            _two_tensor_inputs,
            FUSED_ELEMENTWISE_SHAPE,
            "elementwise_fusion",
            "same_shape",
            tolerance=TaskTolerance(rtol=1e-5, atol=2e-6),
        ),
        _task(
            "add_relu",
            "Fused Add ReLU",
            "Compute relu(x + y) elementwise.",
            add_relu_ref,
            _two_tensor_inputs,
            FUSED_ELEMENTWISE_SHAPE,
            "elementwise_fusion",
            "same_shape",
            tolerance=TaskTolerance(rtol=1e-5, atol=1e-6),
        ),
        _task(
            "residual_add_relu",
            "Fused Residual Add ReLU",
            "Compute relu(x + residual + bias) with bias broadcast over the last dimension.",
            residual_add_relu_ref,
            residual_add_relu_inputs,
            FUSED_ELEMENTWISE_SHAPE,
            "elementwise_fusion",
            "feature_dim",
            tolerance=TaskTolerance(rtol=1e-5, atol=1e-6),
        ),
        _task(
            "bias_gelu",
            "Fused Bias GELU",
            "Compute sigmoid-approximate gelu(x + bias) with bias broadcast over the last dimension.",
            bias_gelu_ref,
            _bias_inputs,
            FUSED_ELEMENTWISE_SHAPE,
            "elementwise_fusion",
            "feature_dim",
            tolerance=TaskTolerance(rtol=2e-4, atol=2e-5),
        ),
        _task(
            "row_sum",
            "Fused Row Sum",
            "Sum a 2D tensor over its final dimension.",
            row_sum_ref,
            row_sum_inputs,
            FUSED_REDUCTION_SHAPE,
            "row_reduction",
            "feature_dim",
            tolerance=TaskTolerance(rtol=1e-4, atol=1e-4),
        ),
        _task(
            "layernorm_small",
            "Fused Small LayerNorm",
            "Apply layer normalization over the final dimension.",
            layernorm_small_ref,
            layernorm_small_inputs,
            FUSED_NORM_SHAPE,
            "row_norm",
            "feature_dim",
            tolerance=TaskTolerance(rtol=2e-4, atol=2e-4),
        ),
        _task(
            "rmsnorm_small",
            "Fused Small RMSNorm",
            "Apply RMS normalization over the final dimension.",
            rmsnorm_small_ref,
            rmsnorm_small_inputs,
            FUSED_NORM_SHAPE,
            "row_norm",
            "feature_dim",
            tolerance=TaskTolerance(rtol=2e-4, atol=2e-4),
        ),
    ]


def _task(
    task_id: str,
    name: str,
    description: str,
    reference_fn,
    input_generator,
    shape: Shape,
    op_class: str,
    shape_kind: str,
    *,
    tolerance: TaskTolerance,
) -> KernelTask:
    return KernelTask(
        task_id=task_id,
        name=name,
        description=description,
        reference_fn=reference_fn,
        input_generator=input_generator,
        allowed_dtypes=(torch.float32,),
        tolerance=tolerance,
        benchmark_shapes=[shape],
        reference_source=_source(reference_fn),
        metadata={
            "task_family": "fused8",
            "op_class": op_class,
            "shape_kind": shape_kind,
            "shape_metadata": {
                "rows": shape[0],
                "feature_dim": shape[1],
                "numel": shape[0] * shape[1],
                "rank": len(shape),
            },
            "prompt_hints": _prompt_hints(task_id),
        },
    )


def _prompt_hints(task_id: str) -> list[str]:
    hints = {
        "bias_relu": [
            "flatten x/output and index bias with offsets % feature_dim",
            "compute tl.maximum(x + bias, 0.0)",
        ],
        "sigmoid_mul": [
            "flatten x/z/output",
            "compute sigmoid as 1 / (1 + exp(-x)) then multiply by z",
        ],
        "add_relu": [
            "flatten x/y/output",
            "compute tl.maximum(x + y, 0.0)",
        ],
        "residual_add_relu": [
            "flatten x/residual/output and index bias by last dimension",
            "compute tl.maximum(x + residual + bias, 0.0)",
        ],
        "bias_gelu": [
            "flatten x/output and index bias by last dimension",
            "use sigmoid GELU approximation to match the reference",
        ],
        "row_sum": [
            "one Triton program per row",
            "reduce the last dimension with tl.sum",
        ],
        "layernorm_small": [
            "one Triton program per row",
            "compute mean and variance over the last dimension, then apply weight and bias",
        ],
        "rmsnorm_small": [
            "one Triton program per row",
            "compute rsqrt(mean(x*x) + eps), then apply weight",
        ],
    }
    return hints.get(task_id, [])
