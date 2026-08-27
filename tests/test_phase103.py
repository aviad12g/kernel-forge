import json
import sys
from pathlib import Path

from openkernelforge.config import load_config
from openkernelforge.datasets.export import export_dataset
from openkernelforge.harness.benchmarker import BenchmarkResult, RuntimeStats
from openkernelforge.reports.final_3task import write_final_3task_report
from openkernelforge.reports.repeatability import write_repeatability_report
from openkernelforge.reports.skipped_variants import write_skipped_variants_artifacts
from openkernelforge.reports.template_report import write_template_autotune_report
from openkernelforge.templates.template_agent import TemplateAgent
from openkernelforge.templates.variant_validation import is_power_of_two, validate_template_variant
from openkernelforge.tasks.simple_tasks import get_task


def test_power_of_two_validator_accepts_valid_block_sizes():
    for block_size in (32, 64, 128):
        result = validate_template_variant({"block_size": block_size})
        assert result.valid
        assert is_power_of_two(block_size)


def test_power_of_two_validator_rejects_invalid_block_sizes():
    for block_size in (48, 96, 192):
        result = validate_template_variant({"block_size": block_size})
        assert not result.valid
        assert result.rejection_reason == "block_size_not_power_of_two_for_tl_arange"
        assert "power of two" in result.warnings[0]


def test_clean_focused_config_loads():
    config = load_config("configs/template_3tasks_gpu_autotune_focused_clean.yaml")
    assert config.agent.type == "template"
    assert config.agent.template_variants["generation_stage"] == "template_focused_clean"
    assert config.agent.template_variants["validate_variants"] is True
    assert config.benchmark.enable_torch_compile is True


def test_clean_template_generation_skips_invalid_variants():
    agent = TemplateAgent(
        template_variants={
            "validate_variants": True,
            "record_skipped_variants": True,
            "block_sizes": [32, 48, 64, 96],
            "num_warps": [1],
            "num_stages": [3],
            "contiguous_policies": ["none"],
            "output_allocation_policies": ["torch.empty"],
            "n_elements_modes": ["runtime"],
            "max_variants_per_task": 10,
        }
    )
    candidates = agent.generate_all(get_task("vector_add"))
    assert [candidate.metadata["block_size"] for candidate in candidates] == [32, 64]
    assert candidates[0].metadata["total_possible_variants"] == 4
    assert candidates[0].metadata["skipped_invalid_variants"] == 2
    assert "block_size_not_power_of_two_for_tl_arange" in candidates[0].metadata["skipped_reasons"]
    assert len(agent.skipped_variants_by_task["vector_add"]) == 2


def test_skipped_variant_counts_appear_in_template_report(tmp_path):
    run_dir = _synthetic_run(tmp_path / "template")
    skipped = [
        {
            "task_id": "vector_add",
            "template_metadata": {"block_size": 48},
            "rejection_reason": "block_size_not_power_of_two_for_tl_arange",
            "warnings": ["power of two"],
            "source_type": "template_variant_validation",
        }
    ]
    write_skipped_variants_artifacts(run_dir, {"vector_add": skipped})
    report = write_template_autotune_report(run_dir)
    text = report.read_text(encoding="utf-8")
    assert "Skipped invalid variants: 1" in text
    assert "block_size_not_power_of_two_for_tl_arange: 1" in text


def test_repeatability_report_generated_with_mocked_benchmark(monkeypatch, tmp_path):
    run_dir = _synthetic_run(tmp_path / "repeatability")
    candidate_modules_before = {
        name for name in sys.modules if name.startswith("openkernelforge_candidate_")
    }

    def fake_benchmark_task(*args, **kwargs):
        return BenchmarkResult(
            task_id="vector_add",
            candidate_name="candidate_000",
            shape=(16,),
            dtype="float32",
            device="cpu",
            eager=RuntimeStats(1.0, 1.0, 1.0, 1.0, [1.0]),
            candidate=RuntimeStats(0.8, 0.8, 0.8, 0.8, [0.8]),
            torch_compile=RuntimeStats(0.9, 0.9, 0.9, 0.9, [0.9]),
            speedup_vs_eager=1.25,
            speedup_vs_torch_compile=1.125,
        )

    monkeypatch.setattr("openkernelforge.reports.repeatability.benchmark_task", fake_benchmark_task)
    report, json_path = write_repeatability_report(run_dir, top_k=1, repeats=3)
    assert report.exists()
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["results"][0]["stats"]["median"] == 1.25
    assert "Repeatability Report" in report.read_text(encoding="utf-8")
    candidate_modules_after = {
        name for name in sys.modules if name.startswith("openkernelforge_candidate_")
    }
    assert candidate_modules_after == candidate_modules_before


def test_final_3task_report_generated_from_synthetic_runs(tmp_path):
    base = _synthetic_run(tmp_path / "base", speedup=0.5)
    shape = _synthetic_run(tmp_path / "shape", speedup=0.8)
    copy = _synthetic_run(tmp_path / "copy", speedup=0.7)
    focused = _synthetic_run(tmp_path / "focused", speedup=0.9)
    clean = _synthetic_run(tmp_path / "clean", speedup=1.1)
    report = write_final_3task_report(
        base_template=base,
        shapeaware=shape,
        template_copy_wide=copy,
        focused=focused,
        clean_focused=clean,
        out=tmp_path / "final.md",
    )
    text = report.read_text(encoding="utf-8")
    assert "Final 3-Task Conclusion" in text
    assert "not a SOTA claim" in text
    assert "bias_relu is the first real above-eager win" in text


def test_dataset_export_excludes_skipped_variants_from_sft_raw(tmp_path):
    run_dir = _synthetic_run(tmp_path / "dataset")
    skipped = [
        {
            "task_id": "vector_add",
            "template_metadata": {"block_size": 48},
            "rejection_reason": "block_size_not_power_of_two_for_tl_arange",
            "warnings": ["power of two"],
            "source_type": "template_variant_validation",
        }
    ]
    write_skipped_variants_artifacts(run_dir, {"vector_add": skipped})
    out_dir = export_dataset(run_dir, tmp_path / "export")
    sft_rows = _read_jsonl(out_dir / "sft_raw.jsonl")
    skipped_rows = _read_jsonl(out_dir / "skipped_variants.jsonl")
    assert len(sft_rows) == 1
    assert len(skipped_rows) == 1
    assert skipped_rows[0]["source_type"] == "template_variant_validation"


def _synthetic_run(run_dir: Path, *, speedup: float = 0.85) -> Path:
    run_dir.mkdir(parents=True)
    candidate_dir = run_dir / "candidates" / "vector_add"
    candidate_dir.mkdir(parents=True)
    candidate_path = candidate_dir / "candidate_000.py"
    candidate_path.write_text(
        "def forward(*args):\n    return args[0]\n",
        encoding="utf-8",
    )
    (run_dir / "environment_probe.json").write_text("{}", encoding="utf-8")
    (run_dir / "config.yaml").write_text(
        "tasks:\n- vector_add\nagent:\n  type: template\nbenchmark:\n  device: cpu\n  warmup: 0\n  repeat: 1\n",
        encoding="utf-8",
    )
    candidate = {
        "record_type": "candidate",
        "task_id": "vector_add",
        "task_name": "Vector Add",
        "candidate_id": "candidate_000",
        "candidate_name": "candidate_000",
        "candidate_path": str(candidate_path),
        "policy_passed": True,
        "verification_passed": True,
        "benchmark_summary": {
            "speedup_vs_eager": speedup,
            "speedup_vs_torch_compile": speedup + 0.1,
            "candidate_median_ms": 1.0,
            "eager_median_ms": speedup,
        },
        "generation_stage": "template_focused_clean",
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
        "variant_validation": {"valid": True, "rejection_reason": None, "warnings": []},
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
