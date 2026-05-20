"""Timing helpers for repeatability-aware benchmarking."""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from openkernelforge.utils.gpu import synchronize_if_cuda


class CacheFlusher(Protocol):
    """Minimal protocol used by timers for optional cache flushing."""

    cache_flush_enabled: bool
    cache_flush_performed: bool

    def flush(self) -> bool:
        ...


@dataclass
class TimingSampleSummary:
    """Summary statistics for a set of runtime samples in milliseconds."""

    n: int
    mean_ms: float | None = None
    median_ms: float | None = None
    p25_ms: float | None = None
    p75_ms: float | None = None
    iqr_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    std_ms: float | None = None
    cv: float | None = None
    bootstrap_ci_low: float | None = None
    bootstrap_ci_high: float | None = None
    samples_ms: list[float] | None = None


def summarize_samples(
    samples: list[float] | tuple[float, ...],
    *,
    bootstrap: bool = False,
    bootstrap_samples: int = 1000,
    seed: int = 123,
) -> TimingSampleSummary:
    """Return robust sample statistics for millisecond timing samples."""

    values = [float(sample) for sample in samples]
    if not values:
        return TimingSampleSummary(n=0, samples_ms=[])

    sorted_values = sorted(values)
    mean = float(statistics.fmean(sorted_values))
    median = float(statistics.median(sorted_values))
    p25 = float(_percentile(sorted_values, 0.25))
    p75 = float(_percentile(sorted_values, 0.75))
    std = float(statistics.stdev(sorted_values)) if len(sorted_values) > 1 else 0.0
    ci_low = None
    ci_high = None
    if bootstrap:
        ci_low, ci_high = _bootstrap_median_ci(
            sorted_values,
            bootstrap_samples=max(1, int(bootstrap_samples)),
            seed=seed,
        )
    return TimingSampleSummary(
        n=len(values),
        mean_ms=mean,
        median_ms=median,
        p25_ms=p25,
        p75_ms=p75,
        iqr_ms=float(p75 - p25),
        min_ms=float(sorted_values[0]),
        max_ms=float(sorted_values[-1]),
        std_ms=std,
        cv=abs(std / mean) if mean else None,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        samples_ms=values,
    )


class WallClockTimer:
    """CPU/dev fallback timer using perf_counter and device synchronization."""

    timing_mode = "wall_clock"

    def measure(
        self,
        fn: Callable[..., Any],
        inputs: tuple[Any, ...],
        *,
        warmup: int,
        repeats: int,
        device: torch.device,
        cache_flusher: CacheFlusher | None = None,
    ) -> list[float]:
        _validate_counts(warmup=warmup, repeats=repeats)
        with torch.no_grad():
            for _ in range(warmup):
                fn(*inputs)
            synchronize_if_cuda(device)

            samples_ms: list[float] = []
            for _ in range(repeats):
                if cache_flusher is not None:
                    cache_flusher.flush()
                synchronize_if_cuda(device)
                start = time.perf_counter()
                fn(*inputs)
                synchronize_if_cuda(device)
                samples_ms.append((time.perf_counter() - start) * 1000.0)
        return samples_ms


class CudaEventTimer:
    """CUDA-event timer for GPU kernel runtime measurements."""

    timing_mode = "cuda_event"

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device or "cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CudaEventTimer requires CUDA, but torch.cuda.is_available() is false")
        if self.device.type != "cuda":
            raise RuntimeError(f"CudaEventTimer requires a CUDA device, got {self.device}")

    def measure(
        self,
        fn: Callable[..., Any],
        inputs: tuple[Any, ...],
        *,
        warmup: int,
        repeats: int,
        device: torch.device,
        cache_flusher: CacheFlusher | None = None,
    ) -> list[float]:
        _validate_counts(warmup=warmup, repeats=repeats)
        if device.type != "cuda":
            raise RuntimeError(f"CUDA-event timing requires a CUDA device, got {device}")

        with torch.no_grad(), torch.cuda.device(device):
            for _ in range(warmup):
                fn(*inputs)
            torch.cuda.synchronize(device)

            samples_ms: list[float] = []
            for _ in range(repeats):
                if cache_flusher is not None:
                    cache_flusher.flush()
                torch.cuda.synchronize(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn(*inputs)
                end.record()
                torch.cuda.synchronize(device)
                samples_ms.append(float(start.elapsed_time(end)))
        return samples_ms


def _validate_counts(*, warmup: int, repeats: int) -> None:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")


def _percentile(sorted_samples: list[float], q: float) -> float:
    if not sorted_samples:
        raise ValueError("Cannot compute percentile of empty samples")
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    pos = (len(sorted_samples) - 1) * q
    lower = int(math.floor(pos))
    upper = min(lower + 1, len(sorted_samples) - 1)
    weight = pos - lower
    return sorted_samples[lower] * (1.0 - weight) + sorted_samples[upper] * weight


def _bootstrap_median_ci(
    values: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    medians: list[float] = []
    n = len(values)
    for _ in range(bootstrap_samples):
        resampled = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(float(statistics.median(resampled)))
    medians.sort()
    return (
        float(_percentile(medians, 0.025)),
        float(_percentile(medians, 0.975)),
    )
