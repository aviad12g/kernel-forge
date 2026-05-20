import json
from pathlib import Path

import torch

from openkernelforge.config import load_config
from openkernelforge.datasets.export import export_dataset
from openkernelforge.reports.fused8 import write_fused8_report
from openkernelforge.tasks.fused_tasks import get_fused_tasks
from openkernelforge.tasks.simple_tasks import get_task
from openkernelforge.templates.template_agent import TemplateAgent
import scripts.compare_fused8 as compare_fused8


FUSED_TASK_IDS = {
    "bias_relu",
    "sigmoid_mul",
    "add_relu",
    "residual_add_relu",
    "bias_gelu",
    "row_sum",
    "layernorm_small",
    "rmsnorm_small",
}


def test_all_fused_tasks_generate_valid_inputs():
    tasks = get_fused_tasks()
    assert {task.task_id for task in tasks} == FUSED_TASK_IDS
    for task in tasks:
        inputs = task.generate_inputs(0, shape=(4, 8), device="cpu")
        assert inputs
        assert all(isinstance(item, torch.Tensor) for item in inputs)
        assert task.metadata["task_family"] == "fused8"


def test_all_fused_task_references_run_on_cpu():
    for task in get_fused_tasks():
        inputs = task.generate_inputs(1, shape=(4, 8), device="cpu")
        output = task.reference_fn(*inputs)
        assert isinstance(output, torch.Tensor)
        assert torch.isfinite(output).all()


def test_fused_task_metadata_includes_shapes_and_tolerances():
    task = get_task("layernorm_small")
    assert task.metadata["shape_metadata"]["feature_dim"] == 1024
    assert task.metadata["prompt_hints"]
    assert task.tolerance.rtol > 0
    assert task.benchmark_shapes == [(4096, 1024)]


def test_fused_template_agent_generates_candidates_for_all_tasks():
    config = load_config("configs/template_fused8_gpu_autotune_quick.yaml")
    agent = TemplateAgent(
        template_family=config.agent.template_family,
        template_variants={**config.agent.template_variants, "max_variants_per_task": 3},
    )
    for task_id in FUSED_TASK_IDS:
        candidates = agent.generate_all(get_task(task_id))
        assert candidates
        assert all(candidate.metadata["template_family"] == "fused8" for candidate in candidates)


def test_fused_template_candidates_expose_forward():
    config = load_config("configs/template_fused8_gpu_autotune_quick.yaml")
    agent = TemplateAgent(
        template_family=config.agent.template_family,
        template_variants={**config.agent.template_variants, "max_variants_per_task": 1},
    )
    for task_id in FUSED_TASK_IDS:
        candidate = agent.generate_all(get_task(task_id))[0]
        assert "def forward(*args):" in candidate.source
        assert "import triton" in candidate.source


def test_fused_configs_load():
    quick = load_config("configs/template_fused8_gpu_autotune_quick.yaml")
    wide = load_config("configs/template_fused8_gpu_autotune_wide.yaml")
    gemini = load_config("configs/gemini_fused8_gpu_baseline.yaml")
    guided = load_config("configs/gemini_fused8_gpu_template_guided.yaml")
    assert quick.agent.template_family == "fused8"
    assert wide.agent.template_variants["max_variants_per_task"] == 300
    assert gemini.agent.type == "llm"
    assert guided.agent.performance_search.include_best_template_context is True


def test_fused8_report_generated_from_synthetic_records(tmp_path):
    run_dir = _synthetic_run(tmp_path / "fused8")
    report = write_fused8_report(run_dir)
    text = report.read_text(encoding="utf-8")
    assert "Fused8 Report" in text
    assert "Task Shapes" in text
    assert "bias_relu" in text


def test_compare_fused8_handles_missing_runs_gracefully(capsys):
    code = compare_fused8.main([])
    captured = capsys.readouterr()
    assert code == 0
    assert "Missing fused8 runs" in captured.out
    assert "template_fused8_gpu_autotune_wide.yaml" in captured.out


def test_dataset_export_includes_task_family_fused8(tmp_path):
    run_dir = _synthetic_run(tmp_path / "dataset")
    out_dir = export_dataset(run_dir, tmp_path / "export")
    rows = _read_jsonl(out_dir / "sft_raw.jsonl")
    assert rows
    assert rows[0]["task_family"] == "fused8"
    assert rows[0]["shape_metadata"]["feature_dim"] == 1024


def _synthetic_run(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True)
    candidate_dir = run_dir / "candidates" / "bias_relu"
    candidate_dir.mkdir(parents=True)
    candidate_path = candidate_dir / "candidate_000.py"
    candidate_path.write_text("import triton\n\ndef forward(*args):\n    return args[0]\n", encoding="utf-8")
    (run_dir / "environment_probe.json").write_text("{}", encoding="utf-8")
    (run_dir / "config.yaml").write_text("agent:\n  type: template\n  template_family: fused8\n", encoding="utf-8")
    candidate = {
        "record_type": "candidate",
        "task_id": "bias_relu",
        "task_name": "Fused Bias ReLU",
        "candidate_id": "candidate_000",
        "candidate_path": str(candidate_path),
        "policy_passed": True,
        "verification_passed": True,
        "benchmark_summary": {
            "speedup_vs_eager": 1.1,
            "speedup_vs_torch_compile": 1.0,
            "candidate_median_ms": 1.0,
            "eager_median_ms": 1.1,
            "torch_compile_median_ms": 1.0,
        },
        "generation_stage": "template_fused8_quick",
        "selected_best": True,
        "template_family": "fused8",
        "task_family": "fused8",
        "template_id": "bias_relu_bs128_nw1_ns3_none_empty_nconstexpr_fconstexpr",
        "block_size": 128,
        "num_warps": 1,
        "num_stages": 3,
        "contiguous_policy": "none",
        "output_allocation_policy": "torch.empty",
        "shape_specialized": True,
        "feature_dim_mode": "constexpr",
        "n_elements_mode": "constexpr",
    }
    task = {
        "record_type": "task_summary",
        "task_id": "bias_relu",
        "task_name": "Fused Bias ReLU",
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
