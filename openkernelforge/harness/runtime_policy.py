"""Runtime guardrails for generated Triton candidates.

Static AST checks are necessary but do not observe dynamically aliased PyTorch
operations. This module audits one pre-timing candidate execution with
TorchDispatchMode and counts Triton JIT launches. It is a policy guardrail, not
an operating-system security boundary.
"""

from __future__ import annotations

import traceback
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from openkernelforge.harness.inputs import clone_inputs
from openkernelforge.tasks.base import KernelTask
from openkernelforge.utils.gpu import synchronize_if_cuda


_ALLOWED_ATEN_PREFIXES = (
    "aten.empty",
    "aten.empty_like",
    "aten.empty_strided",
    "aten.zeros",
    "aten.zeros_like",
    "aten.ones",
    "aten.ones_like",
    "aten.full",
    "aten.full_like",
    "aten.new_empty",
    "aten.new_zeros",
    "aten.new_ones",
    "aten.new_full",
    "aten.detach",
    "aten.alias",
)


@dataclass
class RuntimePolicyResult:
    passed: bool
    observed_aten_ops: list[str] = field(default_factory=list)
    disallowed_aten_ops: list[str] = field(default_factory=list)
    triton_launch_count: int = 0
    error: str | None = None
    policy_version: str = "runtime-v1"


class TorchOperatorAuditMode(TorchDispatchMode):
    """Record ATen operators and fail immediately on compute fallbacks."""

    def __init__(self) -> None:
        super().__init__()
        self.observed: list[str] = []
        self.disallowed: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        name = str(func)
        self.observed.append(name)
        if not any(name.startswith(prefix) for prefix in _ALLOWED_ATEN_PREFIXES):
            self.disallowed.append(name)
            raise RuntimeError(f"runtime policy rejected ATen compute: {name}")
        return func(*args, **(kwargs or {}))


class TritonLaunchCounter(AbstractContextManager["TritonLaunchCounter"]):
    """Count calls to Triton's JIT launch path during one audited execution."""

    def __init__(self) -> None:
        self.count = 0
        self._target: type[Any] | None = None
        self._original: Callable[..., Any] | None = None

    def __enter__(self) -> "TritonLaunchCounter":
        try:
            from triton.runtime.jit import JITFunction
        except Exception as exc:  # pragma: no cover - depends on optional Triton install
            raise RuntimeError("runtime Triton launch audit requires Triton") from exc
        original = getattr(JITFunction, "run", None)
        if not callable(original):
            raise RuntimeError("unsupported Triton runtime: JITFunction.run is unavailable")
        counter = self

        def counted(instance, *args, **kwargs):
            counter.count += 1
            return original(instance, *args, **kwargs)

        self._target = JITFunction
        self._original = original
        setattr(JITFunction, "run", counted)
        return self

    def __exit__(self, exc_type, exc_value, traceback_object) -> bool:
        del exc_type, exc_value, traceback_object
        if self._target is not None and self._original is not None:
            setattr(self._target, "run", self._original)
        return False


def audit_candidate_runtime(
    task: KernelTask,
    candidate: Callable[..., Any],
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> RuntimePolicyResult:
    """Audit one candidate call before benchmark warmup and timing."""

    selected_device = torch.device(device)
    inputs = task.generate_inputs(seed, task.benchmark_shapes[0], dtype, selected_device)
    # Harness preparation is not candidate behavior and must remain outside the
    # TorchDispatchMode policy boundary.
    audited_inputs = clone_inputs(inputs)
    operator_mode = TorchOperatorAuditMode()
    launch_counter = TritonLaunchCounter()
    try:
        with torch.no_grad(), operator_mode, launch_counter:
            candidate(*audited_inputs)
            synchronize_if_cuda(selected_device)
    except Exception:
        return RuntimePolicyResult(
            passed=False,
            observed_aten_ops=operator_mode.observed,
            disallowed_aten_ops=operator_mode.disallowed,
            triton_launch_count=launch_counter.count,
            error=traceback.format_exc(),
        )
    if launch_counter.count < 1:
        return RuntimePolicyResult(
            passed=False,
            observed_aten_ops=operator_mode.observed,
            disallowed_aten_ops=operator_mode.disallowed,
            triton_launch_count=0,
            error="runtime policy observed no Triton JIT launch",
        )
    return RuntimePolicyResult(
        passed=True,
        observed_aten_ops=operator_mode.observed,
        disallowed_aten_ops=operator_mode.disallowed,
        triton_launch_count=launch_counter.count,
    )
