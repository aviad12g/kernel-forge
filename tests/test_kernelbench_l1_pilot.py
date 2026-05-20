from __future__ import annotations

import json
from pathlib import Path

import pytest

from openkernelforge.cli import main
from openkernelforge.config import load_config
from openkernelforge.reports.kernelbench_l1 import run_kernelbench_l1_check, write_kernelbench_l1_report
from openkernelforge.tasks.kernelbench_l1 import (
    KernelBenchL1Error,
    make_candidate_provider,
    load_kernelbench_l1_tasks,
)


def _write_synthetic_task(root: Path, task_id: str = "kb_l1_add") -> Path:
    task_dir = root / "level1"
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"{task_id}.py"
    path.write_text(
        f"""
import torch

TASK_ID = "{task_id}"
TASK_NAME = "Synthetic add"
OP_FAMILY = "elementwise"
BENCHMARK_SHAPES = [(4, 4)]
TOLERANCE = {{"rtol": 1e-4, "atol": 1e-5}}
INPUT_SPEC = {{"shape": [4, 4], "inputs": ["x", "y"]}}

def reference_fn(x, y):
    return x + y

def input_generator(seed, shape, dtype, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    y = torch.randn(shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    return x, y
""",
        encoding="utf-8",
    )
    return path


def _write_synthetic_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
tasks: []
output_dir: {json.dumps(str(output_dir))}
kernelbench:
  max_tasks: 5
  candidate_provider: none
benchmark:
  enabled: true
  timing_mode: wall_clock
  warmup: 0
  repeat: 2
  independent_sessions: 1
  include_torch_compile: false
execution:
  require_cuda: false
  require_triton: false
  require_tiny_triton_kernel: false
""",
        encoding="utf-8",
    )


def test_kernelbench_l1_adapter_loads_synthetic_task_stubs(tmp_path):
    _write_synthetic_task(tmp_path)
    tasks = load_kernelbench_l1_tasks(tmp_path, max_tasks=5)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.task_id == "kb_l1_add"
    assert task.metadata["task_family"] == "kernelbench_l1"
    assert task.metadata["op_family"] == "elementwise"
    inputs = task.generate_inputs(0)
    assert len(inputs) == 2
    assert task.reference_fn(*inputs).shape == inputs[0].shape


def test_kernelbench_l1_adapter_fails_clearly_when_dir_missing(tmp_path):
    with pytest.raises(KernelBenchL1Error, match="KernelBench L1 tasks were not found"):
        load_kernelbench_l1_tasks(tmp_path / "missing")


def test_kernelbench_l1_configs_load():
    five = load_config("configs/kernelbench_l1_5task_rigorous.yaml")
    twenty = load_config("configs/kernelbench_l1_20task_rigorous.yaml")
    assert five.benchmark.timing_mode == "cuda_event"
    assert five.benchmark.cache_flush.enabled is True
    assert five.benchmark.independent_sessions == 3
    assert twenty.benchmark.repeats == 120
    assert twenty.execution.require_cuda is True


def test_kernelbench_l1_check_creates_report_from_synthetic_tasks(tmp_path):
    task_root = tmp_path / "kb"
    _write_synthetic_task(task_root)
    config_path = tmp_path / "config.yaml"
    _write_synthetic_config(config_path, tmp_path / "runs")

    report = run_kernelbench_l1_check(config_path, task_root)

    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "KernelBench L1 Pilot Report" in text
    assert "`kb_l1_add`" in text
    data = json.loads((report.parent / "kernelbench_l1_check.json").read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert data["records"][0]["reference_ok"] is True


def test_kernelbench_l1_report_generator_handles_no_candidate_results(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    data = {
        "run_dir": str(run_dir),
        "status": "completed",
        "kernelbench_dir": "synthetic",
        "tasks_loaded": 1,
        "timing": {"timing_mode": "wall_clock", "cache_flush_enabled": False, "independent_sessions": 1},
        "environment": {"python_version": "test", "platform": "test", "viability": "CPU_ONLY"},
        "records": [
            {
                "task_id": "kb_l1_add",
                "op_family": "elementwise",
                "shape": [4, 4],
                "reference_ok": True,
                "candidate_path": None,
                "benchmark_summary": {"eager_median_ms": 0.1},
            }
        ],
    }
    report = write_kernelbench_l1_report(run_dir, data=data)
    text = report.read_text(encoding="utf-8")
    assert "Candidate generation is intentionally optional" in text
    assert "Single-run wins: none recorded" in text


def test_kernelbench_candidate_provider_llm_later_raises_clear_error():
    provider = make_candidate_provider({"candidate_provider": "llm_later"})
    with pytest.raises(NotImplementedError, match="not implemented"):
        provider.candidate_for_task("task")


def test_kernelbench_l1_cli_check_creates_report(tmp_path):
    task_root = tmp_path / "kb"
    _write_synthetic_task(task_root)
    config_path = tmp_path / "config.yaml"
    _write_synthetic_config(config_path, tmp_path / "runs")

    code = main(
        [
            "kernelbench-l1-check",
            "--config",
            str(config_path),
            "--kernelbench-dir",
            str(task_root),
        ]
    )

    assert code == 0
    reports = list((tmp_path / "runs").glob("*/kernelbench_l1_check.md"))
    assert reports
