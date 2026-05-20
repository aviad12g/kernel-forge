import json
from pathlib import Path

from openkernelforge.agents.code_extract import CodeExtractionResult
from openkernelforge.agents.llm_agent import LLMGeneration, LLMAgent
from openkernelforge.agents.performance import build_performance_prompt
from openkernelforge.config import (
    AgentConfig,
    BenchmarkConfig,
    PerformanceSearchConfig,
    RunConfig,
    VerificationConfig,
    load_config,
)
from openkernelforge.datasets.export import export_dataset
from openkernelforge.harness.template_preservation import check_template_preservation
import openkernelforge.harness.runner as runner_module
from openkernelforge.harness.runner import run_from_config
from openkernelforge.harness.verifier import VerificationResult
from openkernelforge.reports.profiler_lite import write_profiler_lite_report
from openkernelforge.tasks.simple_tasks import get_task
from openkernelforge.templates.elementwise_templates import render_relu_template, render_vector_add_template
import scripts.compare_template_copy as compare_template_copy


def test_profiler_lite_report_generated_from_synthetic_records(tmp_path):
    run_dir = _synthetic_run(tmp_path / "run")
    report = write_profiler_lite_report(run_dir)
    text = report.read_text(encoding="utf-8")
    assert "Profiler-Lite Report" in text
    assert "static heuristics, not hardware-profiler facts" in text
    assert "Fastest Candidates" in text
    assert "Runtime Distribution By Task" in text


def test_strict_template_copy_prompt_includes_constraints():
    template = render_vector_add_template(block_size=256, num_warps=4)
    prompt = build_performance_prompt(
        task=get_task("vector_add"),
        previous_candidate=template,
        benchmark_summary={"speedup_vs_eager": 0.8, "candidate_median_ms": 0.02},
        template_context={
            "candidate_code": template,
            "requested_parameters": {
                "block_size": 512,
                "num_warps": 8,
                "contiguous_policy": "preserve_template",
            },
        },
        performance_prompt_version="v3_strict_template_copy",
    )
    assert "Preserve the template structure exactly" in prompt
    assert "Do not add try/except" in prompt
    assert "Do not add PyTorch fallback" in prompt
    assert "Keep exactly one Triton kernel launch" in prompt
    assert "Keep the same grid logic" in prompt
    assert "Only allowed modifications" in prompt
    assert "vary BLOCK_SIZE if requested" in prompt


def test_template_preservation_checker_passes_near_identical_candidate():
    source = render_relu_template(block_size=128, num_warps=4)
    result = check_template_preservation(source, source, task_id="relu")
    assert result.passed
    assert result.score >= 90


def test_template_preservation_checker_rejects_added_torch_relu():
    template = render_relu_template(block_size=128, num_warps=4)
    candidate = template.replace("return output", "return torch.relu(output)")
    result = check_template_preservation(candidate, template, task_id="relu")
    assert not result.passed
    assert result.rejection_reason == "template_preservation_forbidden_torch_ops"
    assert any("forbidden torch ops" in warning for warning in result.warnings)


def test_template_preservation_checker_rejects_try_except_fallback():
    template = render_relu_template(block_size=128, num_warps=4)
    candidate = template.replace(
        "def forward(*args):",
        "def forward(*args):\n    try:\n        import triton\n    except ImportError:\n        return torch.relu(args[0])",
    )
    result = check_template_preservation(candidate, template, task_id="relu")
    assert not result.passed
    assert result.rejection_reason in {
        "template_preservation_fallback_detected",
        "template_preservation_forbidden_torch_ops",
    }


def test_template_copy_config_loads():
    config = load_config("configs/gemini_3_1_flash_lite_3tasks_gpu_template_copy.yaml")
    assert config.agent.performance_search.enabled
    assert config.agent.performance_search.mode == "template_copy"
    assert config.agent.performance_prompt_version == "v3_strict_template_copy"
    assert config.agent.template_copy.reject_if_preservation_score_below == 70


def test_template_copy_wide_config_loads():
    config = load_config("configs/gemini_3_1_flash_lite_3tasks_gpu_template_copy_wide.yaml")
    assert config.agent.performance_search.mode == "template_copy"
    assert config.agent.performance_search.candidates_per_setting == 2
    assert config.agent.performance_search.max_settings_per_task == 14


def test_template_copy_records_include_preservation_fields(monkeypatch, tmp_path):
    template_run = _synthetic_template_run(tmp_path / "template_run")
    template_source = render_vector_add_template(block_size=128, num_warps=4)

    def fake_generate_template_copy_candidate(self, task, **kwargs):
        return LLMGeneration(
            attempt_index=kwargs["attempt_index"],
            prompt="strict template-copy prompt",
            raw_response=template_source,
            extraction=CodeExtractionResult(code=template_source + "\n"),
            candidate_name=f"fake_template_copy_{task.task_id}",
            metadata={
                "agent": "llm",
                "backend": "fake",
                "stage": "template_copy",
                "prompt_version": self.prompt_version,
                "repair_prompt_version": self.repair_prompt_version,
                "performance_prompt_version": self.performance_prompt_version,
            },
        )

    def fake_load_and_verify(task, candidate_path, candidate_name, config, verification_device):
        return None, VerificationResult(task_id=task.task_id, candidate_name=candidate_name, passed=True), []

    monkeypatch.setattr(LLMAgent, "generate_template_copy_candidate", fake_generate_template_copy_candidate)
    monkeypatch.setattr(runner_module, "_load_and_verify", fake_load_and_verify)
    config = RunConfig(
        tasks=["vector_add"],
        output_dir=str(tmp_path / "runs"),
        agent=AgentConfig(
            type="llm",
            backend="fake",
            allow_torch_fallback=False,
            performance_prompt_version="v3_strict_template_copy",
            performance_search=PerformanceSearchConfig(
                enabled=True,
                mode="template_copy",
                include_best_template_context=True,
                template_run_dir=str(template_run),
                parameter_grid={
                    "block_sizes": [128],
                    "num_warps": [4],
                    "contiguous_policies": ["preserve_template"],
                },
                max_settings_per_task=1,
            ),
        ),
        verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
        benchmark=BenchmarkConfig(enabled=False, device="cpu"),
    )
    run_dir = run_from_config(config)
    candidate = _candidate_records(run_dir)[0]
    assert candidate["generation_stage"] == "template_copy"
    assert candidate["template_source_path"]
    assert candidate["copied_from_template_id"] == "vector_add_bs128_nw4_none"
    assert candidate["requested_block_size"] == 128
    assert candidate["preserved_template_structure_score"] >= 70
    assert candidate["template_preservation"]["passed"]


def test_compare_template_copy_handles_missing_run(capsys, tmp_path):
    template = _synthetic_run(tmp_path / "template")
    guided = _synthetic_run(tmp_path / "guided")
    code = compare_template_copy.main(["--template", str(template), "--template-guided", str(guided)])
    captured = capsys.readouterr()
    assert code == 0
    assert "Missing template-copy run" in captured.out
    assert "gemini_3_1_flash_lite_3tasks_gpu_template_copy.yaml" in captured.out


def test_dataset_export_includes_template_copy_metadata(tmp_path):
    run_dir = _synthetic_run(tmp_path / "template_copy_run", stage="template_copy")
    out_dir = export_dataset(run_dir, tmp_path / "dataset")
    rows = _read_jsonl(out_dir / "sft_raw.jsonl")
    assert rows
    row = rows[0]
    assert row["source_type"] == "template_copy"
    assert row["generation_stage"] == "template_copy"
    assert row["template_source_path"]
    assert row["preserved_template_structure_score"] == 95
    optimization_rows = _read_jsonl(out_dir / "optimization.jsonl")
    assert any(row.get("target_type") == "template_copy_optimization" for row in optimization_rows)


def _synthetic_template_run(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True)
    candidate_dir = run_dir / "candidates" / "vector_add"
    candidate_dir.mkdir(parents=True)
    candidate_path = candidate_dir / "candidate_000.py"
    candidate_path.write_text(render_vector_add_template(block_size=128, num_warps=4), encoding="utf-8")
    (run_dir / "environment_probe.json").write_text("{}", encoding="utf-8")
    (run_dir / "config.yaml").write_text("agent:\n  type: template\n", encoding="utf-8")
    candidate = _candidate_record(candidate_path, stage="template_baseline")
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


def _synthetic_run(run_dir: Path, stage: str = "initial") -> Path:
    run_dir.mkdir(parents=True)
    candidate_dir = run_dir / "candidates" / "vector_add"
    candidate_dir.mkdir(parents=True)
    candidate_path = candidate_dir / "candidate_000.py"
    candidate_path.write_text(render_vector_add_template(block_size=128, num_warps=4), encoding="utf-8")
    template_path = candidate_dir / "template.py"
    template_path.write_text(render_vector_add_template(block_size=256, num_warps=4), encoding="utf-8")
    (run_dir / "environment_probe.json").write_text("{}", encoding="utf-8")
    (run_dir / "config.yaml").write_text("agent:\n  type: llm\n  backend: fake\n", encoding="utf-8")
    candidate = _candidate_record(candidate_path, stage=stage, template_path=template_path)
    task = {
        "record_type": "task_summary",
        "task_id": "vector_add",
        "task_name": "Vector Add",
        "verification": {"passed": True},
        "benchmarks": [],
        "candidate_records": [candidate],
    }
    (run_dir / "results.jsonl").write_text(
        json.dumps(candidate) + "\n" + json.dumps(task) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _candidate_record(candidate_path: Path, *, stage: str, template_path: Path | None = None) -> dict:
    record = {
        "record_type": "candidate",
        "task_id": "vector_add",
        "task_name": "Vector Add",
        "candidate_id": "candidate_000",
        "candidate_path": str(candidate_path),
        "policy_passed": True,
        "verification_passed": True,
        "benchmark_summary": {
            "speedup_vs_eager": 0.75,
            "candidate_median_ms": 1.0,
            "eager_median_ms": 0.75,
        },
        "benchmark_result": [
            {
                "candidate": {
                    "median_ms": 1.0,
                    "mean_ms": 1.1,
                    "p25_ms": 0.9,
                    "p75_ms": 1.2,
                }
            }
        ],
        "generation_stage": stage,
        "agent_type": "llm",
        "backend": "fake",
        "selected_best": True,
    }
    if stage == "template_baseline":
        record.update(
            {
                "template_family": "elementwise",
                "template_id": "vector_add_bs128_nw4_none",
                "block_size": 128,
                "num_warps": 4,
                "contiguous_policy": "none",
            }
        )
    if stage == "template_copy":
        record.update(
            {
                "template_source_path": str(template_path or candidate_path),
                "copied_from_template_id": "vector_add_bs256_nw4_none",
                "requested_block_size": 128,
                "requested_num_warps": 4,
                "requested_contiguous_policy": "preserve_template",
                "preserved_template_structure_score": 95,
                "template_preservation": {"passed": True, "score": 95, "warnings": []},
                "extra_torch_ops_detected": False,
                "fallback_detected": False,
                "source_template_benchmark": {"speedup_vs_eager": 0.7, "candidate_median_ms": 1.1},
                "source_template_speedup_vs_eager": 0.7,
                "delta_vs_source_template": 0.05,
            }
        )
    return record


def _candidate_records(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("record_type") == "candidate"
    ]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
