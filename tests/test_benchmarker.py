import pytest
import torch

from openkernelforge.harness.benchmarker import benchmark_task
from openkernelforge.tasks.simple_tasks import get_task


def test_benchmarker_returns_positive_runtime():
    task = get_task("vector_add")

    def forward(x, y):
        return x + y

    result = benchmark_task(
        task,
        forward,
        shape=(16,),
        device="cpu",
        dtype="float32",
        warmup=1,
        repeats=3,
        enable_torch_compile=False,
    )
    assert result.benchmark_error is None
    assert result.eager is not None
    assert result.candidate is not None
    assert result.eager.median_ms > 0
    assert result.candidate.median_ms > 0


def test_benchmarker_prepares_callables_once_and_rotates_order():
    task = get_task("relu")

    class Prepared:
        def __init__(self):
            self.prepares = 0

        def prepare_for(self, dtype, device):
            self.prepares += 1
            return torch.relu

        def __call__(self, x):
            raise AssertionError("unprepared callable should not execute")

    candidate = Prepared()
    result = benchmark_task(
        task,
        candidate,
        shape=(8,),
        device="cpu",
        warmup=0,
        repeats=1,
        independent_sessions=3,
    )
    assert result.benchmark_error is None
    assert candidate.prepares == 1
    assert [row["measurement_order"] for row in result.session_summaries] == [
        ["eager", "candidate"],
        ["candidate", "eager"],
        ["eager", "candidate"],
    ]


def test_benchmarker_reports_session_level_bootstrap_summary():
    task = get_task("relu")
    result = benchmark_task(
        task,
        torch.relu,
        shape=(8,),
        device="cpu",
        warmup=0,
        repeats=2,
        independent_sessions=3,
        bootstrap_ci_config={"enabled": True, "samples": 50, "seed": 7},
    )
    assert result.session_speedup_summary is not None
    assert result.session_speedup_summary["n"] == 3
    assert result.session_speedup_summary["bootstrap_ci_low"] is not None
    assert "fewer than five same-process sessions" in result.measurement_warnings[0]


def test_benchmarker_rejects_invalid_counts():
    task = get_task("relu")
    with pytest.raises(ValueError, match="independent_sessions must be positive"):
        benchmark_task(task, torch.relu, device="cpu", independent_sessions=0)


def test_torch_compile_time_includes_first_materializing_call(monkeypatch):
    task = get_task("relu")
    calls = []

    def fake_compile(fn, **kwargs):
        def compiled(*args):
            calls.append("compiled_call")
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    result = benchmark_task(
        task,
        torch.relu,
        shape=(8,),
        device="cpu",
        warmup=0,
        repeats=1,
        independent_sessions=1,
        enable_torch_compile=True,
        separate_compile_time=True,
    )
    assert result.compile_error is None
    assert result.compile_time_ms is not None
    assert result.compile_time_kind == "wrapper_and_first_call"
    assert len(calls) >= 2
