import json

from openkernelforge.agents.backends import FakeBackend
from openkernelforge.agents.llm_agent import LLMAgent
from openkernelforge.config import AgentConfig, BenchmarkConfig, RunConfig, VerificationConfig
from openkernelforge.harness.benchmarker import BenchmarkResult, RuntimeStats
import openkernelforge.harness.runner as runner_module
from openkernelforge.harness.runner import run_from_config
from openkernelforge.harness.sandbox import load_candidate_from_path
from openkernelforge.harness.verifier import verify_candidate
from openkernelforge.tasks.simple_tasks import get_task


def test_llm_agent_succeeds_with_correct_fake_backend(tmp_path):
    task = get_task("vector_add")
    agent = LLMAgent(FakeBackend(mode="correct"), backend_name="fake")
    generation = agent.generate_initial(task)
    assert generation.extraction.ok

    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text(generation.extraction.code or "", encoding="utf-8")
    loaded = load_candidate_from_path(candidate_path)
    result = verify_candidate(
        task,
        loaded.forward,
        seeds=[0],
        shapes=[(16,)],
        device="cpu",
        dtype="float32",
    )
    assert result.passed


def test_llm_runner_repairs_broken_fake_backend(tmp_path):
    config = RunConfig(
        tasks=["vector_add"],
        output_dir=str(tmp_path),
        agent=AgentConfig(type="llm", backend="fake", fake_mode="broken_then_fixed", max_attempts=3),
        verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
        benchmark=BenchmarkConfig(
            enabled=True,
            warmup=1,
            repeats=2,
            device="cpu",
            max_shapes_per_task=1,
            enable_torch_compile=False,
        ),
    )
    run_dir = run_from_config(config)
    records = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    record = [item for item in records if item.get("record_type") == "task_summary"][0]
    assert record["verification"]["passed"]
    assert len(record["attempts"]) == 2
    assert not record["attempts"][0]["verification"]["passed"]
    assert record["attempts"][1]["verification"]["passed"]
    assert record["attempts"][0]["failure_reason"] == "values_not_close"
    assert (run_dir / "prompts" / "vector_add" / "candidate_001_prompt.txt").exists()
    repair_prompt = (run_dir / "prompts" / "vector_add" / "candidate_001_prompt.txt").read_text(
        encoding="utf-8"
    )
    assert "Verifier feedback" in repair_prompt
    assert "values_not_close" in repair_prompt


def test_llm_runner_saves_prompt_response_candidate_and_attempt_metadata(tmp_path):
    config = RunConfig(
        tasks=["vector_add"],
        output_dir=str(tmp_path),
        agent=AgentConfig(type="llm", backend="fake", fake_mode="correct", max_attempts=3),
        verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
        benchmark=BenchmarkConfig(
            enabled=True,
            warmup=1,
            repeats=2,
            device="cpu",
            max_shapes_per_task=1,
            enable_torch_compile=False,
        ),
    )
    run_dir = run_from_config(config)
    records = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    record = [item for item in records if item.get("record_type") == "task_summary"][0]
    attempt = record["attempts"][0]

    assert record["agent_type"] == "llm"
    assert record["verification"]["passed"]
    assert len(record["attempts"]) == 1
    assert attempt["prompt_path"]
    assert attempt["response_path"]
    assert attempt["candidate_path"]
    assert attempt["extraction"]["metadata"]["has_forward"]
    assert record["benchmarks"]
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "prompts" / "vector_add" / "candidate_000_prompt.txt").exists()
    assert (run_dir / "responses" / "vector_add" / "candidate_000_response.txt").exists()
    assert (run_dir / "candidates" / "vector_add" / "candidate_000.py").exists()


def test_llm_runner_saves_multiple_candidates_per_attempt(tmp_path):
    config = RunConfig(
        tasks=["vector_add"],
        output_dir=str(tmp_path),
        agent=AgentConfig(
            type="llm",
            backend="fake",
            fake_mode="broken_then_fixed",
            max_attempts=3,
            candidates_per_attempt=2,
        ),
        verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
        benchmark=BenchmarkConfig(
            enabled=True,
            warmup=1,
            repeats=2,
            device="cpu",
            max_shapes_per_task=1,
            enable_torch_compile=False,
        ),
    )
    run_dir = run_from_config(config)
    records = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    candidate_records = [record for record in records if record.get("record_type") == "candidate"]
    task_record = [record for record in records if record.get("record_type") == "task_summary"][0]

    assert len(candidate_records) == 2
    assert [record["candidate_index"] for record in candidate_records] == [0, 1]
    assert not candidate_records[0]["verification_passed"]
    assert candidate_records[1]["verification_passed"]
    assert candidate_records[1]["selected_best"]
    assert task_record["verification"]["passed"]
    assert (run_dir / "responses" / "vector_add" / "candidate_001_response.txt").exists()
    assert (run_dir / "candidates" / "vector_add" / "candidate_001.py").exists()


def test_llm_runner_selects_fastest_correct_candidate(monkeypatch, tmp_path):
    def fake_benchmark_task(task, candidate_forward, **kwargs):
        name = kwargs["candidate_name"]
        candidate_ms = 5.0 if name.endswith("c000") else 1.0
        result = BenchmarkResult(
            task_id=task.task_id,
            candidate_name=name,
            shape=(16,),
            dtype="float32",
            device="cpu",
            eager=RuntimeStats(10.0, 10.0, 10.0, 10.0, [10.0]),
            candidate=RuntimeStats(candidate_ms, candidate_ms, candidate_ms, candidate_ms, [candidate_ms]),
            speedup_vs_eager=10.0 / candidate_ms,
        )
        return result

    monkeypatch.setattr(runner_module, "benchmark_task", fake_benchmark_task)
    config = RunConfig(
        tasks=["vector_add"],
        output_dir=str(tmp_path),
        agent=AgentConfig(
            type="llm",
            backend="fake",
            fake_mode="correct",
            max_attempts=1,
            candidates_per_attempt=2,
            benchmark_all_correct=True,
        ),
        verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
        benchmark=BenchmarkConfig(enabled=True, device="cpu", max_shapes_per_task=1),
    )
    run_dir = run_from_config(config)
    records = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    candidate_records = [record for record in records if record.get("record_type") == "candidate"]
    selected = [record for record in candidate_records if record["selected_best"]]
    assert len(selected) == 1
    assert selected[0]["candidate_id"] == "candidate_001"


def test_llm_runner_does_not_select_failed_candidate(tmp_path):
    config = RunConfig(
        tasks=["vector_add"],
        output_dir=str(tmp_path),
        agent=AgentConfig(type="llm", backend="fake", fake_mode="always_broken", max_attempts=1),
        verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
        benchmark=BenchmarkConfig(enabled=False, device="cpu", max_shapes_per_task=1),
    )
    run_dir = run_from_config(config)
    records = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    candidate_records = [record for record in records if record.get("record_type") == "candidate"]
    task_record = [record for record in records if record.get("record_type") == "task_summary"][0]

    assert not task_record["verification"]["passed"]
    assert candidate_records
    assert not any(record["selected_best"] for record in candidate_records)
