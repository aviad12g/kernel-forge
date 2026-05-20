"""Execution-environment probing for CUDA/Triton kernel runs."""

from __future__ import annotations

import dataclasses
import platform as platform_module
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any


CPU_ONLY = "CPU_ONLY"
CUDA_NO_TRITON = "CUDA_NO_TRITON"
TRITON_IMPORT_ONLY = "TRITON_IMPORT_ONLY"
TRITON_EXECUTION_OK = "TRITON_EXECUTION_OK"
UNKNOWN_BROKEN = "UNKNOWN_BROKEN"


@dataclass
class CudaDeviceInfo:
    index: int
    name: str
    capability: str | None = None


@dataclass
class EnvironmentProbeResult:
    python_version: str
    platform: str
    torch_available: bool = False
    torch_version: str | None = None
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_devices: list[CudaDeviceInfo] = field(default_factory=list)
    triton_available: bool = False
    triton_version: str | None = None
    tiny_triton_kernel_passed: bool = False
    viability: str = UNKNOWN_BROKEN
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def probe_environment() -> EnvironmentProbeResult:
    """Probe whether the current process can verify and benchmark Triton kernels."""

    result = EnvironmentProbeResult(
        python_version=sys.version.replace("\n", " "),
        platform=f"{platform_module.system()} {platform_module.release()} ({platform_module.machine()})",
    )

    torch_module = None
    try:
        import torch

        torch_module = torch
        result.torch_available = True
        result.torch_version = str(getattr(torch, "__version__", "unknown"))
        result.cuda_available = bool(torch.cuda.is_available())
        result.cuda_device_count = int(torch.cuda.device_count()) if result.cuda_available else 0
        for index in range(result.cuda_device_count):
            capability = None
            try:
                capability_tuple = torch.cuda.get_device_capability(index)
                capability = ".".join(str(part) for part in capability_tuple)
            except Exception as exc:  # pragma: no cover - device-specific
                result.warnings.append(f"Could not read CUDA capability for device {index}: {exc}")
            try:
                name = torch.cuda.get_device_name(index)
            except Exception as exc:  # pragma: no cover - device-specific
                name = "unknown"
                result.warnings.append(f"Could not read CUDA device name for device {index}: {exc}")
            result.cuda_devices.append(CudaDeviceInfo(index=index, name=name, capability=capability))
    except Exception as exc:
        result.errors.append(f"torch import failed: {exc}")
        result.viability = UNKNOWN_BROKEN
        return result

    try:
        import triton

        result.triton_available = True
        result.triton_version = str(getattr(triton, "__version__", "unknown"))
    except Exception as exc:
        result.errors.append(f"triton import failed: {exc}")

    if not result.cuda_available:
        result.viability = CPU_ONLY
        result.warnings.append("CUDA is not available; Triton kernels cannot be verified or benchmarked here.")
        return result

    if not result.triton_available:
        result.viability = CUDA_NO_TRITON
        result.warnings.append("CUDA is available, but Triton is not importable.")
        return result

    result.tiny_triton_kernel_passed = _run_tiny_triton_kernel(torch_module, result)
    if result.tiny_triton_kernel_passed:
        result.viability = TRITON_EXECUTION_OK
    else:
        result.viability = TRITON_IMPORT_ONLY
        result.warnings.append("Triton imports, but a tiny Triton kernel did not execute successfully.")
    return result


def format_environment_summary(result: EnvironmentProbeResult | dict[str, Any]) -> str:
    """Return a readable, secret-free environment summary."""

    data = result.to_dict() if isinstance(result, EnvironmentProbeResult) else dict(result)
    devices = data.get("cuda_devices") or []
    if devices:
        device_text = ", ".join(
            f"{device.get('index')}: {device.get('name')} sm_{str(device.get('capability') or 'unknown').replace('.', '')}"
            for device in devices
            if isinstance(device, dict)
        )
    else:
        device_text = "none"
    lines = [
        "OpenKernelForge environment check",
        f"- Viability: {data.get('viability', UNKNOWN_BROKEN)}",
        f"- Python: {data.get('python_version', 'n/a')}",
        f"- Platform: {data.get('platform', 'n/a')}",
        f"- Torch: {'yes' if data.get('torch_available') else 'no'} ({data.get('torch_version') or 'n/a'})",
        f"- CUDA: {'yes' if data.get('cuda_available') else 'no'}",
        f"- CUDA devices: {data.get('cuda_device_count', 0)} [{device_text}]",
        f"- Triton: {'yes' if data.get('triton_available') else 'no'} ({data.get('triton_version') or 'n/a'})",
        f"- Tiny Triton kernel: {'pass' if data.get('tiny_triton_kernel_passed') else 'not passed'}",
    ]
    warnings = data.get("warnings") or []
    errors = data.get("errors") or []
    if warnings:
        lines.append("- Warnings:")
        lines.extend(f"  - {warning}" for warning in warnings)
    if errors:
        lines.append("- Errors:")
        lines.extend(f"  - {error}" for error in errors)
    return "\n".join(lines)


def environment_warning_for_triton_run(
    environment: EnvironmentProbeResult | dict[str, Any] | None,
    *,
    requires_triton_kernels: bool,
) -> str | None:
    if not environment or not requires_triton_kernels:
        return None
    data = environment.to_dict() if isinstance(environment, EnvironmentProbeResult) else environment
    viability = data.get("viability")
    if viability in {CPU_ONLY, CUDA_NO_TRITON, TRITON_IMPORT_ONLY, UNKNOWN_BROKEN}:
        return (
            "This run can test model generation and policy behavior, but cannot "
            "verify/benchmark Triton kernels on this machine."
        )
    return None


def _run_tiny_triton_kernel(torch: Any, result: EnvironmentProbeResult) -> bool:
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _tiny_add_one(x_ptr, y_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(y_ptr + offsets, x + 1.0, mask=mask)

        x = torch.arange(16, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        _tiny_add_one[(1,)](x, y, x.numel(), BLOCK=16)
        torch.cuda.synchronize()
        return bool(torch.allclose(y.cpu(), x.cpu() + 1.0))
    except Exception:
        result.errors.append("tiny triton kernel failed:\n" + traceback.format_exc(limit=8))
        return False
