import json

from openkernelforge.config import BenchmarkConfig, RunConfig, VerificationConfig
from openkernelforge.harness.runner import run_from_config


def test_smoke_runner_creates_results_and_summary(tmp_path):
    config = RunConfig(
        tasks=["vector_add", "relu"],
        output_dir=str(tmp_path),
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
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.md"

    assert results_path.exists()
    assert summary_path.exists()

    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    task_records = [record for record in records if record.get("record_type") == "task_summary"]
    candidate_records = [record for record in records if record.get("record_type") == "candidate"]
    assert len(task_records) == 2
    assert len(candidate_records) == 2
    assert sum(1 for record in task_records if record["verification"]["passed"]) == 2
    assert all(record["selected_best"] for record in candidate_records)
