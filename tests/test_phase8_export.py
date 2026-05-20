import json
from pathlib import Path

from openkernelforge.config import AgentConfig, BenchmarkConfig, RunConfig, VerificationConfig
from openkernelforge.datasets.export import export_dataset, validate_dataset
from openkernelforge.harness.benchmarker import BenchmarkResult, RuntimeStats
import openkernelforge.harness.runner as runner_module
from openkernelforge.harness.runner import run_from_config
from openkernelforge.reports.analyze import analyze_run


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_analyze_run_creates_analysis_md_from_fake_run(tmp_path):
    run_dir = run_from_config(
        RunConfig(
            tasks=["vector_add"],
            output_dir=str(tmp_path / "runs"),
            agent=AgentConfig(type="llm", backend="fake", fake_mode="correct"),
            verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
            benchmark=BenchmarkConfig(enabled=False, device="cpu"),
        )
    )
    analysis_path = analyze_run(run_dir)
    text = analysis_path.read_text(encoding="utf-8")
    assert "OpenKernelForge Run Analysis" in text
    assert "Harness-only data: yes" in text


def test_export_dataset_creates_sft_rejected_and_manifest_files(tmp_path):
    run_dir = run_from_config(
        RunConfig(
            tasks=["vector_add"],
            output_dir=str(tmp_path / "runs"),
            agent=AgentConfig(
                type="llm",
                backend="fake",
                fake_mode="correct",
                allow_torch_fallback=False,
                max_attempts=1,
                candidates_per_attempt=1,
            ),
            verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
            benchmark=BenchmarkConfig(enabled=False, device="cpu"),
        )
    )
    out_dir = export_dataset(run_dir, tmp_path / "dataset")
    assert (out_dir / "sft_raw.jsonl").exists()
    assert (out_dir / "rejected.jsonl").exists()
    assert (out_dir / "manifest.json").exists()
    assert _read_jsonl(out_dir / "rejected.jsonl")
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_kind"] == "harness_only"


def test_export_dataset_creates_repair_pair_from_failed_then_correct(tmp_path):
    run_dir = run_from_config(
        RunConfig(
            tasks=["vector_add"],
            output_dir=str(tmp_path / "runs"),
            agent=AgentConfig(
                type="llm",
                backend="fake",
                fake_mode="broken_then_fixed",
                max_attempts=3,
                candidates_per_attempt=1,
                allow_torch_fallback=True,
            ),
            verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
            benchmark=BenchmarkConfig(enabled=False, device="cpu"),
        )
    )
    out_dir = export_dataset(run_dir, tmp_path / "dataset")
    repair_rows = _read_jsonl(out_dir / "repair.jsonl")
    assert repair_rows
    assert repair_rows[0]["broken_code"]
    assert repair_rows[0]["target"]


def test_export_dataset_creates_optimization_pair_from_slow_to_fast(monkeypatch, tmp_path):
    def fake_benchmark_task(task, candidate_forward, **kwargs):
        name = kwargs["candidate_name"]
        candidate_ms = 10.0 if name.endswith("c000") else 1.0
        return BenchmarkResult(
            task_id=task.task_id,
            candidate_name=name,
            shape=(16,),
            dtype="float32",
            device="cpu",
            eager=RuntimeStats(10.0, 10.0, 10.0, 10.0, [10.0]),
            candidate=RuntimeStats(candidate_ms, candidate_ms, candidate_ms, candidate_ms, [candidate_ms]),
            speedup_vs_eager=10.0 / candidate_ms,
        )

    monkeypatch.setattr(runner_module, "benchmark_task", fake_benchmark_task)
    run_dir = run_from_config(
        RunConfig(
            tasks=["vector_add"],
            output_dir=str(tmp_path / "runs"),
            agent=AgentConfig(
                type="llm",
                backend="fake",
                fake_mode="correct",
                max_attempts=1,
                candidates_per_attempt=2,
                benchmark_all_correct=True,
                allow_torch_fallback=True,
            ),
            verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
            benchmark=BenchmarkConfig(enabled=True, device="cpu"),
        )
    )
    out_dir = export_dataset(run_dir, tmp_path / "dataset")
    optimization_rows = _read_jsonl(out_dir / "optimization.jsonl")
    assert optimization_rows
    assert optimization_rows[0]["slow_code"]
    assert optimization_rows[0]["fast_code"]


def test_validate_dataset_passes_on_generated_dataset(tmp_path):
    run_dir = run_from_config(
        RunConfig(
            tasks=["vector_add"],
            output_dir=str(tmp_path / "runs"),
            agent=AgentConfig(type="llm", backend="fake", fake_mode="correct"),
            verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
            benchmark=BenchmarkConfig(enabled=False, device="cpu"),
        )
    )
    out_dir = export_dataset(run_dir, tmp_path / "dataset")
    ok, errors = validate_dataset(out_dir)
    assert ok, errors


def test_validate_dataset_fails_on_malformed_jsonl(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for filename in ["sft_raw.jsonl", "repair.jsonl", "optimization.jsonl", "rejected.jsonl"]:
        (dataset / filename).write_text("", encoding="utf-8")
    (dataset / "sft_raw.jsonl").write_text("{bad json}\n", encoding="utf-8")
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "counts_by_file": {
                    "sft_raw.jsonl": 0,
                    "repair.jsonl": 0,
                    "optimization.jsonl": 0,
                    "rejected.jsonl": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    (dataset / "README.md").write_text("dataset", encoding="utf-8")
    ok, errors = validate_dataset(dataset)
    assert not ok
    assert any("invalid JSON" in error for error in errors)
