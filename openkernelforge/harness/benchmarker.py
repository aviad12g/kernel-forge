"""Runtime benchmarking for reference and candidate implementations."""

from __future__ import annotations

import statistics
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from openkernelforge.tasks.base import KernelTask, Shape
from openkernelforge.utils.gpu import dtype_from_name, resolve_device, synchronize_if_cuda

_TORCH_COMPILE_CACHE: dict[tuple[int, str], Callable[..., Any]] = {}


@dataclass
class RuntimeStats:
    median_ms: float
    mean_ms: float
    p25_ms: float
    p75_ms: float
    samples_ms: list[float] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    task_id: str
    candidate_name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    eager: RuntimeStats | None = None
    candidate: RuntimeStats | None = None
    torch_compile: RuntimeStats | None = None
    speedup_vs_eager: float | None = None
    speedup_vs_torch_compile: float | None = None
    compile_error: str | None = None
    benchmark_error: str | None = None


def benchmark_task(
    task: KernelTask,
    candidate_forward: Callable[..., Any],
    *,
    candidate_name: str = "candidate",
    shape: Shape | None = None,
    dtype: str | torch.dtype | None = None,
    device: str | torch.device = "auto",
    warmup: int = 5,
    repeats: int = 20,
    enable_torch_compile: bool = False,
) -> BenchmarkResult:
    selected_shape = shape or task.benchmark_shapes[0]
    selected_dtype = dtype_from_name(dtype) if isinstance(dtype, str) else (dtype or task.allowed_dtypes[0])
    selected_device = resolve_device(device)
    dtype_name = str(selected_dtype).replace("torch.", "")

    result = BenchmarkResult(
        task_id=task.task_id,
        candidate_name=candidate_name,
        shape=tuple(selected_shape),
        dtype=dtype_name,
        device=str(selected_device),
    )

    try:
        base_inputs = task.generate_inputs(1234, selected_shape, selected_dtype, selected_device)
        eager_inputs = _clone_inputs(base_inputs)
        candidate_inputs = _clone_inputs(base_inputs)
        result.eager = _time_callable(
            task.reference_fn,
            eager_inputs,
            warmup=warmup,
            repeats=repeats,
            device=selected_device,
        )
        result.candidate = _time_callable(
            candidate_forward,
            candidate_inputs,
            warmup=warmup,
            repeats=repeats,
            device=selected_device,
        )
        if result.eager and result.candidate and result.candidate.median_ms > 0:
            result.speedup_vs_eager = result.eager.median_ms / result.candidate.median_ms

        if enable_torch_compile:
            try:
                result.torch_compile = _benchmark_torch_compile(
                    task.reference_fn,
                    _clone_inputs(base_inputs),
                    warmup=warmup,
                    repeats=repeats,
                    device=selected_device,
                )
            except Exception:
                result.compile_error = traceback.format_exc()
            if (
                result.torch_compile
                and result.candidate
                and result.candidate.median_ms > 0
            ):
                result.speedup_vs_torch_compile = (
                    result.torch_compile.median_ms / result.candidate.median_ms
                )
    except Exception:
        result.benchmark_error = traceback.format_exc()

    return result


def _benchmark_torch_compile(
    fn: Callable[..., Any],
    inputs: tuple[Any, ...],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> RuntimeStats | None:
    if not hasattr(torch, "compile"):
        return None
    cache_key = (id(fn), str(device))
    compiled = _TORCH_COMPILE_CACHE.get(cache_key)
    if compiled is None:
        compiled = torch.compile(fn)
        _TORCH_COMPILE_CACHE[cache_key] = compiled
    return _time_callable(compiled, inputs, warmup=warmup, repeats=repeats, device=device)


def _clone_inputs(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    cloned: list[Any] = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            cloned.append(item.clone())
        else:
            cloned.append(item)
    return tuple(cloned)


def _time_callable(
    fn: Callable[..., Any],
    inputs: tuple[Any, ...],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> RuntimeStats:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    with torch.no_grad():
        for _ in range(warmup):
            fn(*inputs)
        synchronize_if_cuda(device)

        samples_ms: list[float] = []
        for _ in range(repeats):
            synchronize_if_cuda(device)
            start = time.perf_counter()
            fn(*inputs)
            synchronize_if_cuda(device)
            samples_ms.append((time.perf_counter() - start) * 1000.0)

    return _stats(samples_ms)


def _stats(samples_ms: list[float]) -> RuntimeStats:
    sorted_samples = sorted(samples_ms)
    return RuntimeStats(
        median_ms=float(statistics.median(sorted_samples)),
        mean_ms=float(statistics.fmean(sorted_samples)),
        p25_ms=float(_percentile(sorted_samples, 0.25)),
        p75_ms=float(_percentile(sorted_samples, 0.75)),
        samples_ms=[float(x) for x in samples_ms],
    )


def _percentile(sorted_samples: list[float], q: float) -> float:
    if not sorted_samples:
        raise ValueError("Cannot compute percentile of empty samples")
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    pos = (len(sorted_samples) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_samples) - 1)
    weight = pos - lower
    return sorted_samples[lower] * (1.0 - weight) + sorted_samples[upper] * weight
