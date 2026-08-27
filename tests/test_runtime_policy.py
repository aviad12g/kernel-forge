from __future__ import annotations

import pytest
import torch

import openkernelforge.harness.runtime_policy as runtime_policy
from openkernelforge.harness.runtime_policy import TorchOperatorAuditMode


def test_runtime_policy_allows_allocation_ops() -> None:
    mode = TorchOperatorAuditMode()
    with mode:
        value = torch.empty((4,), dtype=torch.float32)
    assert value.shape == (4,)
    assert mode.observed
    assert not mode.disallowed


def test_runtime_policy_rejects_high_level_compute() -> None:
    mode = TorchOperatorAuditMode()
    with pytest.raises(RuntimeError, match="ATen compute"):
        with mode:
            torch.ones(4) + torch.ones(4)
    assert any(name.startswith("aten.add") for name in mode.disallowed)


def test_runtime_audit_excludes_harness_input_cloning(monkeypatch) -> None:
    class FakeTask:
        benchmark_shapes = [{"n": 4}]

        @staticmethod
        def generate_inputs(seed, shape, dtype, device):
            del seed, shape, device
            return (torch.arange(4, dtype=dtype),)

    class FakeLaunchCounter:
        def __init__(self) -> None:
            self.count = 0

        def __enter__(self):
            self.count = 1
            return self

        def __exit__(self, exc_type, exc_value, traceback_object):
            del exc_type, exc_value, traceback_object
            return False

    monkeypatch.setattr(runtime_policy, "TritonLaunchCounter", FakeLaunchCounter)
    monkeypatch.setattr(runtime_policy, "synchronize_if_cuda", lambda device: None)

    result = runtime_policy.audit_candidate_runtime(
        FakeTask(),
        lambda value: torch.empty_like(value),
        seed=0,
        dtype=torch.float32,
        device="cpu",
    )

    assert result.passed
    assert not any(name.startswith("aten.clone") for name in result.observed_aten_ops)
