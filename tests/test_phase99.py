import json
from pathlib import Path

import pytest

from openkernelforge.agents.performance import build_performance_prompt
from openkernelforge.config import (
    AgentConfig,
    BenchmarkConfig,
    PerformanceSearchConfig,
    RunConfig,
    VerificationConfig,
    load_config,
)
from openkernelforge.harness.benchmarker import BenchmarkResult, RuntimeStats
import openkernelforge.harness.runner as runner_module
from openkernelforge.harness.runner import run_from_config
from openkernelforge.harness.verifier import VerificationResult
from openkernelforge.reports.template_report import write_template_autotune_report
from openkernelforge.tasks.simple_tasks import get_task
from openkernelforge.templates.elementwise_templates import (
    render_bias_relu_template,
    render_relu_template,
    render_vector_add_template,
)
from openkernelforge.templates.template_agent import TemplateAgent
import scripts.compare_llm_vs_templates as compare_llm_vs_templates


def test_vector_add_template_contains_triton_kernel_and_forward():
    source = render_vector_add_template(block_size=1024, num_warps=4)
    assert "import triton" in source
    assert "@triton.jit" in source
    assert "def forward(*args):" in source
    assert "x_vals + y_vals" in source
    compile(source, "candidate.py", "exec")


def test_relu_template_uses_tl_maximum_and_no_torch_relu_in_forward():
    source = render_relu_template(block_size=512, num_warps=8)
    forward_source = source.split("def forward(*args):", 1)[1]
    assert "tl.maximum" in source
    assert "torch.relu" not in forward_source
    compile(source, "candidate.py", "exec")


def test_bias_relu_template_uses_modulo_bias_indexing():
    source = render_bias_relu_template(block_size=256, num_warps=4)
    assert "feature_idx = offsets % feature_dim" in source
    assert "bias_ptr + feature_idx" in source
    assert "tl.maximum(x_vals + bias_vals, 0.0)" in source
    compile(source, "candidate.py", "exec")


def test_template_agent_generates_expected_number_of_variants():
    agent = TemplateAgent(
        template_variants={
            "block_sizes": [128, 256],
            "num_warps": [4, 8],
            "contiguous_policies": ["none", "wrapper_contiguous"],
        }
    )
    candidates = agent.generate_all(get_task("vector_add"))
    assert len(candidates) == 8
    assert {candidate.metadata["block_size"] for candidate in candidates} == {128, 256}
    assert {candidate.metadata["num_warps"] for candidate in candidates} == {4, 8}


def test_template_config_loads():
    config = load_config("configs/template_3tasks_gpu_autotune.yaml")
    assert config.agent.type == "template"
    assert config.agent.template_family == "elementwise"
    assert config.agent.template_variants["block_sizes"] == [128, 256, 512, 1024, 2048]


def test_template_candidate_records_include_template_metadata(monkeypatch, tmp_path):
    def fake_load_and_verify(task, candidate_path, candidate_name, config, verification_device):
        return None, VerificationResult(task_id=task.task_id, candidate_name=candidate_name, passed=True), []

    monkeypatch.setattr(runner_module, "_load_and_verify", fake_load_and_verify)
    config = RunConfig(
        tasks=["vector_add"],
        output_dir=str(tmp_path / "runs"),
        agent=AgentConfig(
            type="template",
            allow_torch_fallback=False,
            template_variants={
                "block_sizes": [128],
                "num_warps": [4],
                "contiguous_policies": ["none"],
            },
        ),
        verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
        benchmark=BenchmarkConfig(enabled=False, device="cpu"),
    )
    run_dir = run_from_config(config)
    candidate = _candidate_records(run_dir)[0]
    assert candidate["generation_stage"] == "template_baseline"
    assert candidate["template_family"] == "elementwise"
    assert candidate["block_size"] == 128
    assert candidate["num_warps"] == 4
    assert candidate["contiguous_policy"] == "none"
    assert candidate["template_id"]


def test_compare_llm_vs_templates_handles_missing_template_run(capsys, tmp_path):
    llm = _synthetic_run(tmp_path / "llm", "llm")
    code = compare_llm_vs_templates.main(["--llm", str(llm)])
    captured = capsys.readouterr()
    assert code == 0
    assert "Missing template run" in captured.out
    assert "configs/template_3tasks_gpu_autotune.yaml" in captured.out


def test_template_report_can_be_generated_from_synthetic_records(tmp_path):
    run_dir = _synthetic_run(tmp_path / "template", "template")
    path = write_template_autotune_report(run_dir)
    text = path.read_text(encoding="utf-8")
    assert "Template Autotune Report" in text
    assert "Per-Task Leaderboards" in text
    assert "BLOCK_SIZE" in text


def test_template_guided_prompt_includes_best_template_code():
    prompt = build_performance_prompt(
        task=get_task("vector_add"),
        previous_candidate="def forward(*args): return args[0]",
        benchmark_summary={"speedup_vs_eager": 0.5},
        template_context={
            "template_id": "vector_add_bs1024_nw4_none",
            "block_size": 1024,
            "num_warps": 4,
            "contiguous_policy": "none",
            "benchmark_summary": {"speedup_vs_eager": 1.2},
            "candidate_code": "import triton\n\ndef forward(*args):\n    pass",
        },
        performance_prompt_version="v2_template_guided_perf",
    )
    assert "Best deterministic template context" in prompt
    assert "vector_add_bs1024_nw4_none" in prompt
    assert "Improve or adapt this known-correct template" in prompt
    assert "import triton" in prompt


def test_template_guided_config_fails_clearly_if_template_run_dir_missing(monkeypatch, tmp_path):
    def fake_benchmark_task(task, candidate_forward, **kwargs):
        return BenchmarkResult(
            task_id=task.task_id,
            candidate_name=kwargs["candidate_name"],
            shape=(16,),
            dtype="float32",
            device="cpu",
            eager=RuntimeStats(10.0, 10.0, 10.0, 10.0, [10.0]),
            candidate=RuntimeStats(20.0, 20.0, 20.0, 20.0, [20.0]),
            speedup_vs_eager=0.5,
        )

    monkeypatch.setattr(runner_module, "benchmark_task", fake_benchmark_task)
    config = RunConfig(
        tasks=["vector_add"],
        output_dir=str(tmp_path / "runs"),
        agent=AgentConfig(
            type="llm",
            backend="fake",
            fake_mode="correct",
            allow_torch_fallback=True,
            max_attempts=1,
            candidates_per_attempt=1,
            performance_search=PerformanceSearchConfig(
                enabled=True,
                max_rounds=1,
                candidates_per_round=1,
                target_speedup_vs_eager=1.0,
                include_best_template_context=True,
                template_run_dir=str(tmp_path / "missing_template_run"),
            ),
        ),
        verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
        benchmark=BenchmarkConfig(enabled=True, device="cpu", max_shapes_per_task=1),
    )
    with pytest.raises(RuntimeError, match="Run template autotune first|Template run directory"):
        run_from_config(config)


def _candidate_records(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("record_type") == "candidate"
    ]


def _synthetic_run(run_dir: Path, kind: str) -> Path:
    run_dir.mkdir(parents=True)
    candidate_dir = run_dir / "candidates" / "vector_add"
    candidate_dir.mkdir(parents=True)
    candidate_path = candidate_dir / "candidate_000.py"
    candidate_path.write_text("import torch\n\ndef forward(*args):\n    return args[0]\n", encoding="utf-8")
    (run_dir / "environment_probe.json").write_text(
        json.dumps({"viability": "TRITON_EXECUTION_OK", "cuda_available": True, "triton_available": True}),
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(f"agent:\n  type: {kind}\n", encoding="utf-8")
    candidate = {
        "record_type": "candidate",
        "task_id": "vector_add",
        "task_name": "Vector Add",
        "candidate_id": "candidate_000",
        "candidate_path": str(candidate_path),
        "policy_passed": True,
        "verification_passed": True,
        "benchmark_summary": {"speedup_vs_eager": 0.75, "candidate_median_ms": 1.0},
        "generation_stage": "template_baseline" if kind == "template" else "initial",
        "template_family": "elementwise" if kind == "template" else None,
        "template_id": "vector_add_bs128_nw4_none" if kind == "template" else None,
        "block_size": 128 if kind == "template" else None,
        "num_warps": 4 if kind == "template" else None,
        "contiguous_policy": "none" if kind == "template" else None,
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
