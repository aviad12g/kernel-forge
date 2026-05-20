import json
from pathlib import Path

import pytest

from openkernelforge.config import load_config
from openkernelforge.datasets.export import export_dataset
from openkernelforge.harness import benchmarker as benchmarker_module
from openkernelforge.harness.benchmarker import benchmark_task
from openkernelforge.reports.focused_sweep import write_focused_sweep_report
from openkernelforge.tasks.simple_tasks import get_task
from openkernelforge.templates.template_agent import TemplateAgent
import scripts.compare_focused_sweep as compare_focused_sweep


def test_focused_config_loads():
    config = load_config("configs/template_3tasks_gpu_autotune_focused.yaml")
    assert config.agent.type == "template"
    assert config.agent.template_variants["generation_stage"] == "template_focused_sweep"
    assert config.benchmark.enable_torch_compile is True


def test_per_task_template_grids_are_parsed_correctly():
    config = load_config("configs/template_3tasks_gpu_autotune_focused.yaml")
    grids = config.agent.template_variants["per_task"]
    assert grids["vector_add"]["block_sizes"] == [32, 48, 64, 96, 128, 192, 256]
    assert grids["relu"]["output_allocation_policies"] == ["torch.empty", "torch.empty_like"]
    assert grids["bias_relu"]["feature_dim_modes"] == ["constexpr", "runtime"]


def test_focused_variant_generation_includes_vector_add_metadata():
    config = load_config("configs/template_3tasks_gpu_autotune_focused.yaml")
    agent = TemplateAgent(template_variants=config.agent.template_variants)
    candidates = agent.generate_all(get_task("vector_add"))
    assert len(candidates) == 252
    assert candidates[0].metadata["generation_stage"] == "template_focused_sweep"
    assert candidates[0].metadata["focused_seed_candidate_path"].endswith(
        "candidates/vector_add/candidate_120.py"
    )
    assert any(
        candidate.metadata["block_size"] == 64
        and candidate.metadata["num_warps"] == 2
        and candidate.metadata["num_stages"] == 3
        and candidate.metadata["n_elements_mode"] == "constexpr"
        for candidate in candidates
    )


def test_focused_variant_generation_includes_bias_relu_constexpr_metadata():
    config = load_config("configs/template_3tasks_gpu_autotune_focused.yaml")
    agent = TemplateAgent(template_variants=config.agent.template_variants)
    candidates = agent.generate_all(get_task("bias_relu"))
    assert len(candidates) == 500
    assert candidates[0].metadata["total_possible_variants"] == 756
    assert candidates[0].metadata["grid_was_capped"] is True
    assert any(
        candidate.metadata["feature_dim_mode"] == "constexpr"
        and candidate.metadata["n_elements_mode"] == "constexpr"
        and candidate.metadata["shape_specialized"] is True
        for candidate in candidates
    )


def test_torch_compile_failure_is_recorded_cleanly(monkeypatch):
    def failing_compile(fn):
        raise RuntimeError("mock torch.compile failure")

    monkeypatch.setattr(benchmarker_module.torch, "compile", failing_compile, raising=False)
    benchmarker_module._TORCH_COMPILE_CACHE.clear()

    task = get_task("relu")

    def forward(x):
        return x

    result = benchmark_task(
        task,
        forward,
        shape=(8,),
        device="cpu",
        dtype="float32",
        warmup=0,
        repeats=1,
        enable_torch_compile=True,
    )
    assert result.benchmark_error is None
    assert result.compile_error is not None
    assert "mock torch.compile failure" in result.compile_error
    assert result.torch_compile is None


def test_focused_sweep_report_generated_from_synthetic_records(tmp_path):
    focused = _synthetic_run(tmp_path / "focused", stage="template_focused_sweep", speedup=0.97)
    shapeaware = _synthetic_run(tmp_path / "shapeaware", speedup=0.95)
    copy = _synthetic_run(tmp_path / "copy", speedup=0.90)
    report = write_focused_sweep_report(
        focused,
        shapeaware_run=shapeaware,
        template_copy_wide_run=copy,
    )
    text = report.read_text(encoding="utf-8")
    assert "Focused Sweep Report" in text
    assert "Per-Task Top 20" in text
    assert "Sensitivity Tables" in text
    assert "Focused best compile" in text


def test_compare_focused_sweep_handles_missing_focused_run_gracefully(capsys, tmp_path):
    shapeaware = _synthetic_run(tmp_path / "shapeaware")
    copy = _synthetic_run(tmp_path / "copy")
    code = compare_focused_sweep.main(
        ["--shapeaware", str(shapeaware), "--template-copy-wide", str(copy)]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "Missing focused sweep run" in captured.out
    assert "template_3tasks_gpu_autotune_focused.yaml" in captured.out


def test_dataset_export_includes_template_focused_sweep_metadata(tmp_path):
    run_dir = _synthetic_run(tmp_path / "focused", stage="template_focused_sweep", speedup=0.99)
    out_dir = export_dataset(run_dir, tmp_path / "dataset")
    rows = _read_jsonl(out_dir / "sft_raw.jsonl")
    assert rows
    row = rows[0]
    assert row["source_type"] == "template"
    assert row["generation_stage"] == "template_focused_sweep"
    assert row["focused_seed_run"] == "/workspace/openkernelforge/runs/seed"
    assert row["focused_seed_candidate_path"].endswith("candidate_seed.py")
    assert row["speedup_vs_compile"] == pytest.approx(1.05)


def _synthetic_run(
    run_dir: Path,
    *,
    stage: str = "template_baseline",
    speedup: float = 0.85,
) -> Path:
    run_dir.mkdir(parents=True)
    candidate_dir = run_dir / "candidates" / "vector_add"
    candidate_dir.mkdir(parents=True)
    candidate_path = candidate_dir / "candidate_000.py"
    candidate_path.write_text("import triton\n\ndef forward(*args):\n    return args[0]\n", encoding="utf-8")
    (run_dir / "environment_probe.json").write_text("{}", encoding="utf-8")
    (run_dir / "config.yaml").write_text("agent:\n  type: template\n", encoding="utf-8")
    candidate = {
        "record_type": "candidate",
        "task_id": "vector_add",
        "task_name": "Vector Add",
        "candidate_id": "candidate_000",
        "candidate_path": str(candidate_path),
        "policy_passed": True,
        "verification_passed": True,
        "benchmark_summary": {
            "speedup_vs_eager": speedup,
            "speedup_vs_torch_compile": 1.05,
            "candidate_median_ms": 1.0,
            "eager_median_ms": speedup,
            "torch_compile_median_ms": 1.05,
        },
        "generation_stage": stage,
        "selected_best": True,
        "template_family": "elementwise",
        "template_id": "vector_add_bs64_nw2_ns3_none_empty_nconstexpr_fn/a",
        "block_size": 64,
        "num_warps": 2,
        "num_stages": 3,
        "contiguous_policy": "none",
        "output_allocation_policy": "torch.empty",
        "shape_specialized": True,
        "feature_dim_mode": "n/a",
        "n_elements_mode": "constexpr",
        "focused_seed_run": "/workspace/openkernelforge/runs/seed",
        "focused_seed_candidate_path": "/workspace/openkernelforge/runs/seed/candidates/vector_add/candidate_seed.py",
        "focused_rank": 1,
    }
    task = {
        "record_type": "task_summary",
        "task_id": "vector_add",
        "task_name": "Vector Add",
        "verification": {"passed": True},
        "candidate_records": [candidate],
    }
    (run_dir / "results.jsonl").write_text(
        json.dumps(candidate) + "\n" + json.dumps(task) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
