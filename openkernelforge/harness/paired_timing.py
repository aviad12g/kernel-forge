"""Randomized paired CUDA-event timing for process-isolated confirmation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from openkernelforge.harness.cache_flush import CacheFlushConfig, CudaCacheFlusher
from openkernelforge.harness.inputs import clone_inputs
from openkernelforge.tasks.base import KernelTask


@dataclass(frozen=True)
class PairedTimingConfig:
    blocks: int = 20
    warmup_launches: int = 30
    minimum_interval_ms: float = 1.5
    maximum_launches_per_interval: int = 65_536
    seed: int = 0
    cache_l2_multiplier: float = 2.0
    cache_minimum_size_mb: int = 32
    cache_maximum_size_mb: int = 512
    cache_mode: str = "read_write"


@dataclass
class PairedTimingResult:
    task_id: str
    process_id: str
    seed: int
    method_launch_counts: dict[str, int]
    cache_buffer_mb: int
    l2_size_bytes: int
    blocks: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)


def configure_precision_settings(
    *,
    allow_tf32_matmul: bool,
    allow_tf32_cudnn: bool,
    float32_matmul_precision: str,
) -> dict[str, Any]:
    """Set and return the explicit precision policy used by benchmark workers."""

    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32_matmul)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32_cudnn)
    torch.set_float32_matmul_precision(str(float32_matmul_precision))
    return {
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def benchmark_paired_blocks(
    task: KernelTask,
    methods: Mapping[str, Callable[..., Any]],
    *,
    process_id: str,
    config: PairedTimingConfig,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> PairedTimingResult:
    selected_device = torch.device(device)
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("paired confirmation requires CUDA")
    if len(methods) < 2:
        raise ValueError("paired timing requires at least two methods")
    if config.blocks <= 0 or config.warmup_launches < 0:
        raise ValueError("invalid paired timing counts")

    prepared = {
        name: _prepare_callable(fn, dtype, selected_device)
        for name, fn in methods.items()
    }
    l2_size = detected_l2_size_bytes(selected_device)
    cache_mb = cache_buffer_size_mb(
        l2_size,
        multiplier=config.cache_l2_multiplier,
        minimum_mb=config.cache_minimum_size_mb,
        maximum_mb=config.cache_maximum_size_mb,
    )
    flusher = CudaCacheFlusher(
        CacheFlushConfig(enabled=True, size_mb=cache_mb, mode=config.cache_mode),
        device=selected_device,
    )

    warmup_inputs = task.generate_inputs(
        config.seed,
        task.benchmark_shapes[0],
        dtype,
        selected_device,
    )
    with torch.no_grad():
        for name in sorted(prepared):
            for _ in range(config.warmup_launches):
                prepared[name](*clone_inputs(warmup_inputs))
        torch.cuda.synchronize(selected_device)

    launch_counts = {
        name: calibrate_launch_count(
            fn,
            clone_inputs(warmup_inputs),
            device=selected_device,
            minimum_interval_ms=config.minimum_interval_ms,
            maximum_launches=config.maximum_launches_per_interval,
        )
        for name, fn in prepared.items()
    }
    result = PairedTimingResult(
        task_id=task.task_id,
        process_id=str(process_id),
        seed=config.seed,
        method_launch_counts=launch_counts,
        cache_buffer_mb=cache_mb,
        l2_size_bytes=l2_size,
        environment=cuda_environment_snapshot(selected_device),
    )
    rng = random.Random(config.seed)
    with torch.no_grad():
        for block_index in range(config.blocks):
            input_seed = config.seed + 1000 + block_index
            base_inputs = task.generate_inputs(
                input_seed,
                task.benchmark_shapes[0],
                dtype,
                selected_device,
            )
            method_order = list(prepared)
            rng.shuffle(method_order)
            timings: dict[str, float] = {}
            for method_name in method_order:
                flusher.flush()
                timings[method_name] = measure_cuda_interval_ms(
                    prepared[method_name],
                    clone_inputs(base_inputs),
                    launches=launch_counts[method_name],
                    device=selected_device,
                )
            result.blocks.append(
                {
                    "block_id": str(block_index),
                    "input_seed": input_seed,
                    "input_snapshot_sha256": hash_input_snapshot(base_inputs),
                    "method_order": method_order,
                    "median_ms_per_launch": timings,
                    "clock_snapshot": cuda_environment_snapshot(selected_device),
                }
            )
    result.environment["clock_snapshot_after"] = cuda_environment_snapshot(selected_device)
    return result


def calibrate_launch_count(
    fn: Callable[..., Any],
    inputs: tuple[Any, ...],
    *,
    device: torch.device,
    minimum_interval_ms: float,
    maximum_launches: int,
) -> int:
    if minimum_interval_ms <= 0 or maximum_launches <= 0:
        raise ValueError("calibration limits must be positive")
    elapsed = measure_cuda_interval_ms(fn, inputs, launches=1, device=device)
    if elapsed <= 0:
        return maximum_launches
    return max(1, min(maximum_launches, int(math.ceil(minimum_interval_ms / elapsed))))


def measure_cuda_interval_ms(
    fn: Callable[..., Any],
    inputs: tuple[Any, ...],
    *,
    launches: int,
    device: torch.device,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(launches):
        fn(*inputs)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / launches


def detected_l2_size_bytes(device: torch.device | str) -> int:
    props = torch.cuda.get_device_properties(torch.device(device))
    for name in ("L2_cache_size", "l2_cache_size"):
        value = getattr(props, name, None)
        if value:
            return int(value)
    raise RuntimeError(
        "CUDA device properties do not report L2 size; pass through a supported "
        "PyTorch build before running the prespecified cache policy"
    )


def cache_buffer_size_mb(
    l2_size_bytes: int,
    *,
    multiplier: float,
    minimum_mb: int,
    maximum_mb: int,
) -> int:
    if l2_size_bytes <= 0 or multiplier <= 0:
        raise ValueError("L2 size and multiplier must be positive")
    if minimum_mb <= 0 or maximum_mb < minimum_mb:
        raise ValueError("invalid cache buffer bounds")
    target_mb = math.ceil(l2_size_bytes * multiplier / (1024 * 1024))
    return max(minimum_mb, min(maximum_mb, target_mb))


def hash_input_snapshot(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def cuda_environment_snapshot(device: torch.device | str) -> dict[str, Any]:
    selected = torch.device(device)
    props = torch.cuda.get_device_properties(selected)
    gpu_uuid = getattr(props, "uuid", None)
    snapshot: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "device_index": selected.index if selected.index is not None else torch.cuda.current_device(),
        "device_name": props.name,
        # PyTorch may expose this as a UUID object rather than JSON-safe text.
        "gpu_uuid": _json_compatible_uuid(gpu_uuid),
        "l2_size_bytes": getattr(props, "L2_cache_size", getattr(props, "l2_cache_size", None)),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    query = [
        "nvidia-smi",
        "--query-gpu=driver_version,clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,persistence_mode",
        "--format=csv,noheader,nounits",
        "-i",
        str(snapshot["device_index"]),
    ]
    try:
        completed = subprocess.run(query, capture_output=True, text=True, check=True, timeout=10)
        values = [item.strip() for item in completed.stdout.strip().split(",")]
        if len(values) == 6:
            snapshot.update(
                {
                    "driver_version": values[0],
                    "graphics_clock_mhz": values[1],
                    "memory_clock_mhz": values[2],
                    "power_draw_w": values[3],
                    "temperature_c": values[4],
                    "persistence_mode": values[5],
                }
            )
    except Exception as exc:
        snapshot["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def _json_compatible_uuid(value: Any) -> str | None:
    return str(value) if value is not None else None


def write_paired_timing_result(path: str | Path, result: PairedTimingResult) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.__dict__, indent=2) + "\n", encoding="utf-8")
    return output


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
