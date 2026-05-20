from openkernelforge.config import AgentConfig, BenchmarkConfig, RunConfig, VerificationConfig
from openkernelforge.harness.runner import run_from_config
from openkernelforge.reports.compare import compare_runs_markdown


def test_compare_runs_outputs_markdown_table(tmp_path):
    first = run_from_config(
        RunConfig(
            tasks=["vector_add"],
            output_dir=str(tmp_path / "a"),
            agent=AgentConfig(type="dummy", allow_torch_fallback=True),
            verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
            benchmark=BenchmarkConfig(enabled=False, device="cpu"),
        )
    )
    second = run_from_config(
        RunConfig(
            tasks=["vector_add"],
            output_dir=str(tmp_path / "b"),
            agent=AgentConfig(type="llm", backend="fake", fake_mode="correct"),
            verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
            benchmark=BenchmarkConfig(enabled=False, device="cpu"),
        )
    )
    table = compare_runs_markdown([first, second])
    assert "| Run dir | Agent/backend/model |" in table
    assert str(first) in table
    assert str(second) in table
    assert "Policy pass rate" in table
