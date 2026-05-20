import json
from pathlib import Path

import pytest
import torch

from openkernelforge.config import load_config
from openkernelforge.harness.benchmarker import benchmark_task
from openkernelforge.harness.cache_flush import CacheFlushConfig, CudaCacheFlusher
from openkernelforge.harness.timing import CudaEventTimer, WallClockTimer, summarize_samples
from openkernelforge.reports.benchmark_methodology import run_benchmark_methodology_check
from openkernelforge.reports.repeatability import classify_repeatability_label
from openkernelforge.tasks.simple_tasks import get_task
from openkernelforge.utils.env_probe import CPU_ONLY, EnvironmentProbeResult


def test_summarize_samples_computes_iqr_and_cv():
    summary = summarize_samples([1.0, 2.0, 3.0, 4.0])
    assert summary.n == 4
    assert summary.median_ms == pytest.approx(2.5)
    assert summary.p25_ms == pytest.approx(1.75)
    assert summary.p75_ms == pytest.approx(3.25)
    assert summary.iqr_ms == pytest.approx(1.5)
    assert summary.cv is not None


def test_bootstrap_ci_is_deterministic_with_seed():
    first = summarize_samples([1.0, 2.0, 3.0], bootstrap=True, bootstrap_samples=50, seed=7)
    second = summarize_samples([1.0, 2.0, 3.0], bootstrap=True, bootstrap_samples=50, seed=7)
    assert first.bootstrap_ci_low == second.bootstrap_ci_low
    assert first.bootstrap_ci_high == second.bootstrap_ci_high


def test_wall_clock_timer_works_on_cpu():
    timer = WallClockTimer()

    def fn(x):
        return x + 1

    samples = timer.measure(fn, (torch.ones(4),), warmup=1, repeats=2, device=torch.device("cpu"))
    assert len(samples) == 2
    assert all(sample >= 0 for sample in samples)


def test_cuda_event_timer_requires_cuda_when_unavailable():
    if torch.cuda.is_available():
        timer = CudaEventTimer(torch.device("cuda"))
        assert timer.timing_mode == "cuda_event"
    else:
        with pytest.raises(RuntimeError, match="requires CUDA"):
            CudaEventTimer(torch.device("cuda"))


def test_cache_flush_noop_on_cpu():
    flusher = CudaCacheFlusher(CacheFlushConfig(enabled=True, size_mb=1), device=torch.device("cpu"))
    assert flusher.flush() is False
    assert flusher.cache_flush_performed is False


def test_benchmark_config_parses_new_timing_fields():
    config = load_config("configs/template_fused8_gpu_benchmark_rigorous.yaml")
    assert config.benchmark.timing_mode == "cuda_event"
    assert config.benchmark.independent_sessions == 3
    assert config.benchmark.cache_flush.enabled is True
    assert config.benchmark.bootstrap_ci.enabled is True


def test_rigorous_small_config_loads():
    config = load_config("configs/template_fused8_gpu_benchmark_rigorous_small.yaml")
    assert config.agent.type == "template"
    assert config.agent.template_family == "fused8"
    assert config.agent.template_variants["max_variants_per_task"] == 20
    assert config.benchmark.timing_mode == "cuda_event"
    assert config.benchmark.warmup == 20
    assert config.benchmark.repeats == 100
    assert config.benchmark.independent_sessions == 3


def test_benchmark_output_includes_timing_metadata():
    task = get_task("relu")

    def forward(x):
        return torch.relu(x)

    result = benchmark_task(
        task,
        forward,
        shape=(8,),
        dtype="float32",
        device="cpu",
        warmup=0,
        repeats=2,
        timing_mode="wall_clock",
        independent_sessions=1,
    )
    assert result.timing_mode == "wall_clock"
    assert result.candidate_ms_summary is not None
    assert result.candidate_ms_summary["median_ms"] is not None
    assert result.eager_ms_summary is not None
    assert result.session_summaries


def test_repeatability_labels_classify_synthetic_runs():
    assert (
        classify_repeatability_label(
            original_speedup=1.2,
            stats={"median": 1.1},
            stable=True,
        )
        == "REPEAT_STABLE_WIN"
    )
    assert (
        classify_repeatability_label(
            original_speedup=1.2,
            stats={"median": 0.9},
            stable=True,
        )
        == "SINGLE_RUN_ONLY_WIN"
    )
    assert (
        classify_repeatability_label(
            original_speedup=0.8,
            stats={"median": 1.05},
            stable=False,
        )
        == "UNSTABLE"
    )
    assert (
        classify_repeatability_label(
            original_speedup=0.8,
            stats={"median": 0.7},
            stable=True,
        )
        == "BELOW_EAGER"
    )
    assert (
        classify_repeatability_label(
            original_speedup=None,
            stats={"median": None},
            stable=False,
        )
        == "INSUFFICIENT_DATA"
    )


def test_methodology_check_handles_missing_cuda_gracefully(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "openkernelforge.reports.benchmark_methodology.probe_environment",
        lambda: EnvironmentProbeResult(
            python_version="test",
            platform="test",
            cuda_available=False,
            viability=CPU_ONLY,
        ),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "tasks: [relu]\noutput_dir: {out}\nagent:\n  type: template\nbenchmark:\n  timing_mode: cuda_event\n".format(
            out=json.dumps(str(tmp_path / "runs"))
        ),
        encoding="utf-8",
    )
    report = run_benchmark_methodology_check(config_path, max_tasks=1)
    assert report.exists()
    assert "Status: skipped" in report.read_text(encoding="utf-8")
