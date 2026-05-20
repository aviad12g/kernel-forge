import json
from pathlib import Path

from openkernelforge.config import load_config
from openkernelforge.datasets.export import export_dataset
from openkernelforge.reports.template_report import write_template_autotune_report
from openkernelforge.tasks.simple_tasks import get_task
from openkernelforge.templates.template_agent import TemplateAgent
import scripts.compare_template_sweeps as compare_template_sweeps


def test_expanded_template_grid_records_num_stages_metadata():
    agent = TemplateAgent(
        template_variants={
            "block_sizes": [64],
            "num_warps": [1, 2],
            "num_stages": [3, 4],
            "contiguous_policies": ["none"],
            "output_allocation_policies": ["torch.empty_like"],
            "max_variants_per_task": 10,
        }
    )
    candidates = agent.generate_all(get_task("vector_add"))
    assert len(candidates) == 4
    assert {candidate.metadata["block_size"] for candidate in candidates} == {64}
    assert {candidate.metadata["num_warps"] for candidate in candidates} == {1, 2}
    assert {candidate.metadata["num_stages"] for candidate in candidates} == {3, 4}


def test_max_variants_per_task_caps_generation_deterministically():
    agent = TemplateAgent(
        template_variants={
            "block_sizes": [64, 128, 256],
            "num_warps": [1, 2],
            "num_stages": [3],
            "contiguous_policies": ["none"],
            "output_allocation_policies": ["torch.empty_like"],
            "max_variants_per_task": 3,
            "grid_sampling": "capped_ordered",
            "sort_order": "small_to_large",
        }
    )
    candidates = agent.generate_all(get_task("relu"))
    assert len(candidates) == 3
    assert candidates[0].metadata["block_size"] == 64
    assert candidates[0].metadata["num_warps"] == 1
    assert candidates[0].metadata["total_possible_variants"] == 6
    assert candidates[0].metadata["actually_generated_variants"] == 3
    assert candidates[0].metadata["grid_was_capped"] is True


def test_shapeaware_bias_relu_candidates_include_feature_dim_metadata():
    agent = TemplateAgent(
        template_variants={
            "block_sizes": [64],
            "num_warps": [1],
            "num_stages": [3],
            "contiguous_policies": ["none"],
            "output_allocation_policies": ["torch.empty_like"],
            "n_elements_modes": ["runtime", "constexpr"],
            "feature_dim_modes": ["generic", "runtime", "constexpr"],
            "max_variants_per_task": 20,
        }
    )
    candidates = agent.generate_all(get_task("bias_relu"))
    assert {candidate.metadata["feature_dim_mode"] for candidate in candidates} == {
        "generic",
        "runtime",
        "constexpr",
    }
    assert any(candidate.metadata["shape_specialized"] for candidate in candidates)


def test_template_quick_config_loads():
    config = load_config("configs/template_3tasks_gpu_autotune_quick.yaml")
    assert config.agent.type == "template"
    assert config.agent.template_variants["max_variants_per_task"] == 50


def test_template_wide_config_loads():
    config = load_config("configs/template_3tasks_gpu_autotune_wide.yaml")
    assert config.agent.template_variants["block_sizes"][-1] == 8192
    assert config.agent.template_variants["num_warps"] == [1, 2, 4, 8]


def test_template_shapeaware_config_loads():
    config = load_config("configs/template_3tasks_gpu_autotune_shapeaware.yaml")
    assert config.agent.template_variants["n_elements_modes"] == ["runtime", "constexpr"]
    assert "constexpr" in config.agent.template_variants["feature_dim_modes"]


def test_template_leaderboard_csv_json_generated_from_synthetic_records(tmp_path):
    run_dir = _synthetic_template_run(tmp_path / "template")
    write_template_autotune_report(run_dir)
    csv_path = run_dir / "template_leaderboard.csv"
    json_path = run_dir / "template_leaderboard.json"
    assert csv_path.exists()
    assert json_path.exists()
    assert "num_stages" in csv_path.read_text(encoding="utf-8").splitlines()[0]
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    assert rows[0]["leaderboard_rank"] == 1
    assert rows[0]["taxonomy_label"]


def test_compare_template_sweeps_handles_missing_runs_gracefully(capsys, tmp_path):
    copy = _synthetic_template_run(tmp_path / "copy")
    code = compare_template_sweeps.main(["--template-copy-wide", str(copy)])
    captured = capsys.readouterr()
    assert code == 0
    assert "Missing Runs" in captured.out
    assert "template_3tasks_gpu_autotune_wide.yaml" in captured.out
    assert "Template Sweep Comparison" in captured.out


def test_dataset_export_includes_expanded_template_metadata(tmp_path):
    run_dir = _synthetic_template_run(tmp_path / "template")
    out_dir = export_dataset(run_dir, tmp_path / "dataset")
    rows = [
        json.loads(line)
        for line in (out_dir / "sft_raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    row = rows[0]
    assert row["source_type"] == "template"
    assert row["num_stages"] == 4
    assert row["output_allocation_policy"] == "torch.empty"
    assert row["shape_specialized"] is True
    assert row["n_elements_mode"] == "constexpr"


def _synthetic_template_run(run_dir: Path) -> Path:
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
        "benchmark_summary": {"speedup_vs_eager": 0.85, "candidate_median_ms": 1.0},
        "generation_stage": "template_baseline",
        "selected_best": True,
        "template_family": "elementwise",
        "template_id": "vector_add_bs64_nw1_ns4_none_empty_nconstexpr_fn/a",
        "block_size": 64,
        "num_warps": 1,
        "num_stages": 4,
        "contiguous_policy": "none",
        "output_allocation_policy": "torch.empty",
        "shape_specialized": True,
        "feature_dim_mode": "n/a",
        "n_elements_mode": "constexpr",
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
