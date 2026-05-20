"""Small built-in PyTorch tasks for smoke testing the harness."""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F

from openkernelforge.tasks.base import KernelTask, Shape, TaskTolerance
from openkernelforge.tasks.fused_tasks import get_fused_tasks


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


def vector_add_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def vector_add_inputs(
    seed: int, shape: Shape, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = _make_generator(seed)
    return _randn(generator, shape, dtype, device), _randn(generator, shape, dtype, device)


def elementwise_mul_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * y


def elementwise_mul_inputs(
    seed: int, shape: Shape, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = _make_generator(seed)
    return _randn(generator, shape, dtype, device), _randn(generator, shape, dtype, device)


def relu_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def relu_inputs(seed: int, shape: Shape, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor]:
    generator = _make_generator(seed)
    return (_randn(generator, shape, dtype, device),)


def bias_relu_ref(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.relu(x + bias)


def bias_relu_inputs(
    seed: int, shape: Shape, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(shape) != 2:
        raise ValueError("bias_relu expects shape (rows, features)")
    generator = _make_generator(seed)
    rows, features = shape
    return (
        _randn(generator, (rows, features), dtype, device),
        _randn(generator, (features,), dtype, device),
    )


def sigmoid_mul_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x) * y


def sigmoid_mul_inputs(
    seed: int, shape: Shape, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = _make_generator(seed)
    return _randn(generator, shape, dtype, device), _randn(generator, shape, dtype, device)


def row_sum_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x, dim=1)


def row_sum_inputs(seed: int, shape: Shape, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor]:
    if len(shape) != 2:
        raise ValueError("row_sum expects shape (rows, cols)")
    generator = _make_generator(seed)
    return (_randn(generator, shape, dtype, device),)


def layernorm_small_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=1e-5)


def layernorm_small_inputs(
    seed: int, shape: Shape, dtype: torch.dtype, device: torch.device
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


def matmul_bias_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x @ weight + bias


def matmul_bias_inputs(
    seed: int, shape: Shape, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(shape) != 3:
        raise ValueError("matmul_bias expects shape (m, k, n)")
    generator = _make_generator(seed)
    m, k, n = shape
    return (
        _randn(generator, (m, k), dtype, device),
        _randn(generator, (k, n), dtype, device),
        _randn(generator, (n,), dtype, device),
    )


def get_builtin_tasks() -> list[KernelTask]:
    float32 = (torch.float32,)
    base_tasks = [
        KernelTask(
            task_id="vector_add",
            name="Vector Add",
            description="Add two same-shaped tensors elementwise.",
            reference_fn=vector_add_ref,
            input_generator=vector_add_inputs,
            allowed_dtypes=float32,
            tolerance=TaskTolerance(rtol=1e-5, atol=1e-6),
            benchmark_shapes=[(1024,), (4096,), (16384,)],
            reference_source=_source(vector_add_ref),
        ),
        KernelTask(
            task_id="elementwise_mul",
            name="Elementwise Multiply",
            description="Multiply two same-shaped tensors elementwise.",
            reference_fn=elementwise_mul_ref,
            input_generator=elementwise_mul_inputs,
            allowed_dtypes=float32,
            tolerance=TaskTolerance(rtol=1e-5, atol=1e-6),
            benchmark_shapes=[(1024,), (4096,), (16384,)],
            reference_source=_source(elementwise_mul_ref),
        ),
        KernelTask(
            task_id="relu",
            name="ReLU",
            description="Apply ReLU to a tensor.",
            reference_fn=relu_ref,
            input_generator=relu_inputs,
            allowed_dtypes=float32,
            tolerance=TaskTolerance(rtol=1e-5, atol=1e-6),
            benchmark_shapes=[(1024,), (4096,), (16384,)],
            reference_source=_source(relu_ref),
        ),
        KernelTask(
            task_id="bias_relu",
            name="Bias ReLU",
            description="Add a 1D bias to a 2D tensor and apply ReLU.",
            reference_fn=bias_relu_ref,
            input_generator=bias_relu_inputs,
            allowed_dtypes=float32,
            tolerance=TaskTolerance(rtol=1e-5, atol=1e-6),
            benchmark_shapes=[(32, 128), (64, 256)],
            reference_source=_source(bias_relu_ref),
        ),
        KernelTask(
            task_id="sigmoid_mul",
            name="Sigmoid Multiply",
            description="Compute sigmoid(x) * y elementwise.",
            reference_fn=sigmoid_mul_ref,
            input_generator=sigmoid_mul_inputs,
            allowed_dtypes=float32,
            tolerance=TaskTolerance(rtol=1e-5, atol=1e-6),
            benchmark_shapes=[(1024,), (4096,), (16384,)],
            reference_source=_source(sigmoid_mul_ref),
        ),
        KernelTask(
            task_id="row_sum",
            name="Row Sum",
            description="Sum a 2D tensor across columns.",
            reference_fn=row_sum_ref,
            input_generator=row_sum_inputs,
            allowed_dtypes=float32,
            tolerance=TaskTolerance(rtol=1e-5, atol=1e-6),
            benchmark_shapes=[(32, 128), (64, 256)],
            reference_source=_source(row_sum_ref),
        ),
        KernelTask(
            task_id="layernorm_small",
            name="Small LayerNorm",
            description="Apply layer normalization over the final dimension of a small 2D tensor.",
            reference_fn=layernorm_small_ref,
            input_generator=layernorm_small_inputs,
            allowed_dtypes=float32,
            tolerance=TaskTolerance(rtol=1e-4, atol=1e-5),
            benchmark_shapes=[(16, 64), (32, 128)],
            reference_source=_source(layernorm_small_ref),
        ),
        KernelTask(
            task_id="matmul_bias",
            name="Matmul Bias",
            description="Compute x @ weight + bias for a small matrix multiplication.",
            reference_fn=matmul_bias_ref,
            input_generator=matmul_bias_inputs,
            allowed_dtypes=float32,
            tolerance=TaskTolerance(rtol=1e-4, atol=1e-5),
            benchmark_shapes=[(16, 32, 16), (32, 64, 32)],
            reference_source=_source(matmul_bias_ref),
        ),
    ]
    fused_tasks = get_fused_tasks()
    fused_ids = {task.task_id for task in fused_tasks}
    return [task for task in base_tasks if task.task_id not in fused_ids] + fused_tasks


def get_task_map() -> dict[str, KernelTask]:
    return {task.task_id: task for task in get_builtin_tasks()}


def get_task(task_id: str) -> KernelTask:
    tasks = get_task_map()
    try:
        return tasks[task_id]
    except KeyError as exc:
        available = ", ".join(sorted(tasks))
        raise KeyError(f"Unknown task '{task_id}'. Available tasks: {available}") from exc
