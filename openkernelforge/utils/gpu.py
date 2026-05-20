"""Device, dtype, and optional Triton helpers."""

from __future__ import annotations

import importlib.util

import torch


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        requested = device.type
    else:
        requested = str(device)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def dtype_from_name(name: str | None) -> torch.dtype:
    if name is None:
        return torch.float32
    normalized = name.replace("torch.", "").lower()
    mapping = {
        "float": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown dtype '{name}'. Allowed names: {allowed}") from exc


def synchronize_if_cuda(device: str | torch.device) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None
