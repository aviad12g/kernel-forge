"""Runtime benchmarking for reference and candidate implementations."""

from __future__ import annotations

import statistics
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from openkernelforge.harness.cache_flush import CacheFlushConfig, CudaCacheFlusher
from openkernelforge.harness.inputs import clone_inputs
from openkernelforge.harness.timing import CudaEventTimer, WallClockTimer, summarize_samples
from openkernelforge.tasks.base import KernelTask, Shape
from openkernelforge.utils.gpu import dtype_from_name, resolve_device

_TORCH_COMPILE_CACHE: dict[tuple[Any, str, str, tuple[Any, ...]], Callable[..., Any]] = {}


@dataclass
class RuntimeStats:
    median_ms: float
    mean_ms: float
    p25_ms: float
    p75_ms: float
    samples_ms: list[float] = field(default_factory=list)
    iqr_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    std_ms: float | None = None
    cv: float | None = None
    bootstrap_ci_low: float | None = None
    bootstrap_ci_high: float | None = None


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
    timing_mode: str = "wall_clock"
    warmup: int = 5
    repeats: int = 20
    independent_sessions: int = 1
    cache_flush_enabled: bool = False
    cache_flush_performed: bool = False
    eager_ms_summary: dict[str, Any] | None = None
    candidate_ms_summary: dict[str, Any] | None = None
    torch_compile_ms_summary: dict[str, Any] | None = None
    compile_time_ms: float | None = None
    compile_time_kind: str | None = None
    runtime_only_ms: float | None = None
    measurement_warnings: list[str] = field(default_factory=list)
    session_summaries: list[dict[str, Any]] = field(default_factory=list)
    across_session_median_speedup: float | None = None
    across_session_iqr: float | None = None
    stable_above_eager: bool | None = None
    stable_above_compile: bool | None = None
    torch_compile_mode: str | None = None
    session_speedup_summary: dict[str, Any] | None = None
    session_compile_speedup_summary: dict[str, Any] | None = None
    session_isolation: str = "same_process"
    measurement_order: str = "rotating"


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
    timing_mode: str = "auto",
    independent_sessions: int = 1,
    cache_flush_config: Any | None = None,
    bootstrap_ci_config: Any | None = None,
    separate_compile_time: bool = True,
    stable_session_threshold: float = 0.98,
    enable_torch_compile: bool = False,
    torch_compile_mode: str | None = None,
) -> BenchmarkResult:
    _validate_benchmark_counts(
        warmup=warmup,
        repeats=repeats,
        independent_sessions=independent_sessions,
    )
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
        timing_mode=_resolve_timing_mode(timing_mode, selected_device),
        warmup=warmup,
        repeats=repeats,
        independent_sessions=int(independent_sessions),
        cache_flush_enabled=bool(_config_value(cache_flush_config, "enabled", False)),
        torch_compile_mode=torch_compile_mode,
    )

    try:
        reference_callable = _prepare_callable(task.reference_fn, selected_dtype, selected_device)
        prepared_candidate = _prepare_callable(candidate_forward, selected_dtype, selected_device)
        session_speedups: list[float] = []
        session_compile_speedups: list[float] = []
        eager_samples_all: list[float] = []
        candidate_samples_all: list[float] = []
        compile_samples_all: list[float] = []

        compiled_callable: Callable[..., Any] | None = None
        if enable_torch_compile:
            try:
                compile_inputs = task.generate_inputs(
                    1234,
                    selected_shape,
                    selected_dtype,
                    selected_device,
                )
                compiled_callable, result.compile_time_ms = _prepare_torch_compile(
                    reference_callable,
                    clone_inputs(compile_inputs),
                    device=selected_device,
                    separate_compile_time=separate_compile_time,
                    torch_compile_mode=torch_compile_mode,
                )
                if separate_compile_time and result.compile_time_ms is not None:
                    result.compile_time_kind = "wrapper_and_first_call"
            except Exception:
                result.compile_error = traceback.format_exc()

        for session_index in range(result.independent_sessions):
            base_inputs = task.generate_inputs(1234 + session_index, selected_shape, selected_dtype, selected_device)
            cache_flusher = _make_cache_flusher(cache_flush_config, selected_device)
            timer = _make_timer(result.timing_mode, selected_device)
            callables: dict[str, Callable[..., Any]] = {
                "eager": reference_callable,
                "candidate": prepared_candidate,
            }
            if compiled_callable is not None:
                callables["compile"] = compiled_callable
            order = _rotated_order(list(callables), session_index)
            measured: dict[str, RuntimeStats] = {}
            for label in order:
                measured[label] = _time_callable(
                    callables[label],
                    clone_inputs(base_inputs),
                    warmup=warmup,
                    repeats=repeats,
                    device=selected_device,
                    timer=timer,
                    cache_flusher=cache_flusher,
                    bootstrap_ci_config=bootstrap_ci_config,
                )
            eager = measured["eager"]
            candidate = measured["candidate"]
            compiled = measured.get("compile")
            result.cache_flush_performed = result.cache_flush_performed or cache_flusher.cache_flush_performed
            if cache_flusher.warning:
                _append_unique(result.measurement_warnings, cache_flusher.warning)

            speedup = eager.median_ms / candidate.median_ms if candidate.median_ms > 0 else None
            compile_speedup = (
                compiled.median_ms / candidate.median_ms
                if compiled is not None and candidate.median_ms > 0
                else None
            )
            if speedup is not None:
                session_speedups.append(float(speedup))
            if compile_speedup is not None:
                session_compile_speedups.append(float(compile_speedup))
            eager_samples_all.extend(eager.samples_ms)
            candidate_samples_all.extend(candidate.samples_ms)
            if compiled:
                compile_samples_all.extend(compiled.samples_ms)

            if session_index == 0:
                result.eager = eager
                result.candidate = candidate
                result.torch_compile = compiled
                result.runtime_only_ms = candidate.median_ms
            result.session_summaries.append(
                {
                    "session_index": session_index,
                    "eager_ms_summary": _stats_to_summary_dict(eager),
                    "candidate_ms_summary": _stats_to_summary_dict(candidate),
                    "torch_compile_ms_summary": _stats_to_summary_dict(compiled) if compiled else None,
                    "speedup_vs_eager": speedup,
                    "speedup_vs_torch_compile": compile_speedup,
                    "cache_flush_performed": cache_flusher.cache_flush_performed,
                    "compile_time_ms": result.compile_time_ms if session_index == 0 else None,
                    "measurement_order": order,
                }
            )

        result.eager_ms_summary = _summary_dict(eager_samples_all, bootstrap_ci_config=bootstrap_ci_config)
        result.candidate_ms_summary = _summary_dict(candidate_samples_all, bootstrap_ci_config=bootstrap_ci_config)
        result.torch_compile_ms_summary = (
            _summary_dict(compile_samples_all, bootstrap_ci_config=bootstrap_ci_config)
            if compile_samples_all
            else None
        )
        if session_speedups:
            result.session_speedup_summary = _summary_dict(
                session_speedups,
                bootstrap_ci_config=bootstrap_ci_config,
            )
            result.across_session_median_speedup = result.session_speedup_summary["median_ms"]
            result.across_session_iqr = result.session_speedup_summary["iqr_ms"]
            result.speedup_vs_eager = (
                result.across_session_median_speedup
                if result.independent_sessions > 1
                else session_speedups[0]
            )
            result.stable_above_eager = _stable_above_threshold(
                session_speedups,
                threshold=stable_session_threshold,
            )
        if session_compile_speedups:
            result.session_compile_speedup_summary = _summary_dict(
                session_compile_speedups,
                bootstrap_ci_config=bootstrap_ci_config,
            )
            result.speedup_vs_torch_compile = (
                result.session_compile_speedup_summary["median_ms"]
                if result.independent_sessions > 1
                else session_compile_speedups[0]
            )
            result.stable_above_compile = _stable_above_threshold(
                session_compile_speedups,
                threshold=stable_session_threshold,
            )
        if _bootstrap_enabled(bootstrap_ci_config) and result.independent_sessions < 5:
            _append_unique(
                result.measurement_warnings,
                "session bootstrap is descriptive with fewer than five same-process sessions",
            )
    except Exception:
        result.benchmark_error = traceback.format_exc()

    return result


def _prepare_torch_compile(
    fn: Callable[..., Any],
    inputs: tuple[Any, ...],
    *,
    device: torch.device,
    separate_compile_time: bool,
    torch_compile_mode: str | None,
) -> tuple[Callable[..., Any] | None, float | None]:
    if not hasattr(torch, "compile"):
        return None, None
    cache_key = _compile_cache_key(fn, device, torch_compile_mode, inputs)
    compiled = _TORCH_COMPILE_CACHE.get(cache_key) if cache_key is not None else None
    compile_time_ms: float | None = None
    if compiled is None:
        start = time.perf_counter()
        if torch_compile_mode:
            compiled = torch.compile(fn, mode=torch_compile_mode)
        else:
            compiled = torch.compile(fn)
        synchronize = device.type == "cuda" and torch.cuda.is_available()
        if synchronize:
            torch.cuda.synchronize(device)
        with torch.no_grad():
            compiled(*inputs)
        if synchronize:
            torch.cuda.synchronize(device)
        if separate_compile_time:
            compile_time_ms = (time.perf_counter() - start) * 1000.0
        if cache_key is not None:
            _TORCH_COMPILE_CACHE[cache_key] = compiled
    return compiled, compile_time_ms


def _time_callable(
    fn: Callable[..., Any],
    inputs: tuple[Any, ...],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
    timer: WallClockTimer | CudaEventTimer | None = None,
    cache_flusher: CudaCacheFlusher | None = None,
    bootstrap_ci_config: Any | None = None,
) -> RuntimeStats:
    timer = timer or WallClockTimer()
    samples_ms = timer.measure(
        fn,
        inputs,
        warmup=warmup,
        repeats=repeats,
        device=device,
        cache_flusher=cache_flusher,
    )

    return _stats(samples_ms, bootstrap_ci_config=bootstrap_ci_config)


def _stats(samples_ms: list[float], *, bootstrap_ci_config: Any | None = None) -> RuntimeStats:
    summary = summarize_samples(
        samples_ms,
        bootstrap=_bootstrap_enabled(bootstrap_ci_config),
        bootstrap_samples=_bootstrap_samples(bootstrap_ci_config),
        seed=_bootstrap_seed(bootstrap_ci_config),
    )
    return RuntimeStats(
        median_ms=float(summary.median_ms or 0.0),
        mean_ms=float(summary.mean_ms or 0.0),
        p25_ms=float(summary.p25_ms or 0.0),
        p75_ms=float(summary.p75_ms or 0.0),
        samples_ms=[float(x) for x in samples_ms],
        iqr_ms=summary.iqr_ms,
        min_ms=summary.min_ms,
        max_ms=summary.max_ms,
        std_ms=summary.std_ms,
        cv=summary.cv,
        bootstrap_ci_low=summary.bootstrap_ci_low,
        bootstrap_ci_high=summary.bootstrap_ci_high,
    )


def _prepare_callable(
    fn: Callable[..., Any],
    dtype: torch.dtype,
    device: torch.device,
) -> Callable[..., Any]:
    prepare = getattr(fn, "prepare_for", None)
    if callable(prepare):
        prepared = prepare(dtype, device)
        if callable(prepared):
            return prepared
    return fn


def _compile_cache_key(
    fn: Callable[..., Any],
    device: torch.device,
    mode: str | None,
    inputs: tuple[Any, ...],
) -> tuple[Any, str, str, tuple[Any, ...]] | None:
    try:
        hash(fn)
    except TypeError:
        return None
    return (fn, str(device), str(mode or "default"), _input_signature(inputs))


def _input_signature(value: Any) -> tuple[Any, ...]:
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype), str(value.device), tuple(value.stride()))
    if isinstance(value, dict):
        return (
            "dict",
            tuple((str(key), _input_signature(child)) for key, child in sorted(value.items(), key=lambda item: str(item[0]))),
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_input_signature(child) for child in value))
    return (type(value).__name__, repr(value))


def _rotated_order(labels: list[str], session_index: int) -> list[str]:
    if not labels:
        return []
    offset = session_index % len(labels)
    return labels[offset:] + labels[:offset]


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _validate_benchmark_counts(*, warmup: int, repeats: int, independent_sessions: int) -> None:
    if int(warmup) < 0:
        raise ValueError("warmup must be non-negative")
    if int(repeats) <= 0:
        raise ValueError("repeats must be positive")
    if int(independent_sessions) <= 0:
        raise ValueError("independent_sessions must be positive")


def _resolve_timing_mode(timing_mode: str, device: torch.device) -> str:
    if timing_mode == "auto":
        return "cuda_event" if device.type == "cuda" and torch.cuda.is_available() else "wall_clock"
    if timing_mode not in {"cuda_event", "wall_clock"}:
        raise ValueError("timing_mode must be 'cuda_event', 'wall_clock', or 'auto'")
    return timing_mode


def _make_timer(timing_mode: str, device: torch.device) -> WallClockTimer | CudaEventTimer:
    if timing_mode == "cuda_event":
        return CudaEventTimer(device)
    return WallClockTimer()


def _make_cache_flusher(config: Any | None, device: torch.device) -> CudaCacheFlusher:
    return CudaCacheFlusher(_cache_flush_config(config), device=device)


def _cache_flush_config(config: Any | None) -> CacheFlushConfig:
    if config is None:
        return CacheFlushConfig()
    if isinstance(config, CacheFlushConfig):
        return config
    if isinstance(config, dict):
        return CacheFlushConfig(**config)
    return CacheFlushConfig(
        enabled=bool(getattr(config, "enabled", False)),
        size_mb=int(getattr(config, "size_mb", 128)),
        mode=str(getattr(config, "mode", "write")),
    )


def _config_value(config: Any | None, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _bootstrap_enabled(config: Any | None) -> bool:
    return bool(_config_value(config, "enabled", False))


def _bootstrap_samples(config: Any | None) -> int:
    return int(_config_value(config, "samples", 1000))


def _bootstrap_seed(config: Any | None) -> int:
    return int(_config_value(config, "seed", 123))


def _summary_dict(samples_ms: list[float], *, bootstrap_ci_config: Any | None) -> dict[str, Any]:
    summary = summarize_samples(
        samples_ms,
        bootstrap=_bootstrap_enabled(bootstrap_ci_config),
        bootstrap_samples=_bootstrap_samples(bootstrap_ci_config),
        seed=_bootstrap_seed(bootstrap_ci_config),
    )
    return {
        "n": summary.n,
        "mean_ms": summary.mean_ms,
        "median_ms": summary.median_ms,
        "p25_ms": summary.p25_ms,
        "p75_ms": summary.p75_ms,
        "iqr_ms": summary.iqr_ms,
        "min_ms": summary.min_ms,
        "max_ms": summary.max_ms,
        "std_ms": summary.std_ms,
        "cv": summary.cv,
        "bootstrap_ci_low": summary.bootstrap_ci_low,
        "bootstrap_ci_high": summary.bootstrap_ci_high,
    }


def _stats_to_summary_dict(stats: RuntimeStats | None) -> dict[str, Any] | None:
    if stats is None:
        return None
    return {
        "n": len(stats.samples_ms),
        "mean_ms": stats.mean_ms,
        "median_ms": stats.median_ms,
        "p25_ms": stats.p25_ms,
        "p75_ms": stats.p75_ms,
        "iqr_ms": stats.iqr_ms,
        "min_ms": stats.min_ms,
        "max_ms": stats.max_ms,
        "std_ms": stats.std_ms,
        "cv": stats.cv,
        "bootstrap_ci_low": stats.bootstrap_ci_low,
        "bootstrap_ci_high": stats.bootstrap_ci_high,
    }


def _stable_above_threshold(values: list[float], *, threshold: float) -> bool:
    if not values:
        return False
    median = statistics.median(values)
    return float(median) >= 1.0 and all(float(value) >= threshold for value in values)
