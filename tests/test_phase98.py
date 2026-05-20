import json
from pathlib import Path

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
from openkernelforge.harness.benchmarker import BenchmarkResult, RuntimeStats
import openkernelforge.harness.runner as runner_module
from openkernelforge.harness.runner import run_from_config
from openkernelforge.reports.performance_search import write_performance_search_report
from openkernelforge.tasks.simple_tasks import get_task
import scripts.compare_perfsearch as compare_perfsearch


def test_performance_search_config_loads():
    config = load_config("configs/gemini_3_1_flash_lite_3tasks_gpu_perfsearch.yaml")
    assert config.agent.performance_search.enabled
    assert config.agent.performance_search.max_rounds == 3
    assert config.agent.performance_search.candidates_per_round == 4
    assert config.agent.performance_prompt_version == "v1_cuda_elementwise_perf"


def test_wide_performance_search_config_loads():
    config = load_config("configs/gemini_3_1_flash_lite_3tasks_gpu_perfsearch_wide.yaml")
    assert config.agent.performance_search.enabled
    assert config.agent.performance_search.max_rounds == 4
    assert config.agent.performance_search.candidates_per_round == 6
    assert config.agent.temperature == 0.5


def test_optimization_prompt_includes_benchmark_feedback_and_task_hints():
    prompt = build_performance_prompt(
        task=get_task("bias_relu"),
        previous_candidate="def forward(x, bias): return x",
        benchmark_summary={
            "candidate_median_ms": 0.08,
            "eager_median_ms": 0.04,
            "speedup_vs_eager": 0.5,
        },
        heuristic_flags=["not using contiguous flattening"],
    )
    assert "previous candidate passed correctness but is too slow" in prompt
    assert "candidate_median_ms: 0.08" in prompt
    assert "speedup_vs_eager: 0.5" in prompt
    assert "feature_idx = offsets % feature_dim" in prompt
    assert "larger BLOCK_SIZE" in prompt
    assert "Return only Python code" in prompt


def test_performance_search_records_include_stage_and_parent_fields(monkeypatch, tmp_path):
    run_dir = _run_perfsearch_with_fake_benchmarks(monkeypatch, tmp_path, target=1.0)
    candidates = _candidate_records(run_dir)
    search = [record for record in candidates if record.get("generation_stage") == "performance_search"]
    assert search
    assert all(record.get("parent_candidate_path") for record in search)
    assert all(record.get("parent_speedup_vs_eager") is not None for record in search)
    assert any(record.get("improved_over_parent") for record in search)


def test_performance_search_stops_when_target_speedup_is_reached(monkeypatch, tmp_path):
    run_dir = _run_perfsearch_with_fake_benchmarks(
        monkeypatch,
        tmp_path,
        target=1.0,
        perf_candidates_per_round=1,
        perf_max_rounds=3,
        perf_speedups=[1.25, 1.25, 1.25],
    )
    search = [
        record
        for record in _candidate_records(run_dir)
        if record.get("generation_stage") == "performance_search"
    ]
    assert len(search) == 1
    assert search[0]["target_reached"]


def test_performance_search_keeps_best_candidate_if_target_not_reached(monkeypatch, tmp_path):
    run_dir = _run_perfsearch_with_fake_benchmarks(
        monkeypatch,
        tmp_path,
        target=2.0,
        perf_candidates_per_round=2,
        perf_max_rounds=1,
        perf_speedups=[0.7, 0.6],
    )
    selected = [record for record in _candidate_records(run_dir) if record.get("selected_best")]
    assert len(selected) == 1
    assert selected[0]["generation_stage"] == "performance_search"
    assert selected[0]["benchmark_summary"]["speedup_vs_eager"] == 0.7
    assert selected[0]["best_final_speedup_vs_eager"] == 0.7


def test_performance_search_report_is_created(monkeypatch, tmp_path):
    run_dir = _run_perfsearch_with_fake_benchmarks(monkeypatch, tmp_path, target=1.0)
    report = write_performance_search_report(run_dir)
    text = report.read_text(encoding="utf-8")
    assert "Performance Search Report" in text
    assert "Generated optimization candidates" in text
    assert "Useful Optimization Pairs" in text


def test_optimization_dataset_rows_include_parent_and_child_benchmarks(monkeypatch, tmp_path):
    run_dir = _run_perfsearch_with_fake_benchmarks(monkeypatch, tmp_path, target=2.0)
    out_dir = export_dataset(run_dir, tmp_path / "dataset")
    rows = _read_jsonl(out_dir / "optimization.jsonl")
    perf_rows = [row for row in rows if row.get("generation_stage") == "performance_search"]
    assert perf_rows
    row = perf_rows[0]
    assert row["parent_benchmark"]
    assert row["child_benchmark"]
    assert row["speedup_delta"] > 0
    assert row["optimization_prompt"]
    assert row["parent_slow_code"]
    assert row["optimized_child_code"]


def test_compare_perfsearch_handles_missing_run(capsys, tmp_path):
    baseline = _synthetic_baseline_run(tmp_path)
    code = compare_perfsearch.main(["--baseline", str(baseline)])
    captured = capsys.readouterr()
    assert code == 0
    assert "Missing performance-search run" in captured.out
    assert "configs/gemini_3_1_flash_lite_3tasks_gpu_perfsearch.yaml" in captured.out


def _run_perfsearch_with_fake_benchmarks(
    monkeypatch,
    tmp_path,
    *,
    target: float,
    perf_candidates_per_round: int = 2,
    perf_max_rounds: int = 1,
    perf_speedups: list[float] | None = None,
) -> Path:
    perf_speedups = perf_speedups or [0.8, 0.6]

    def fake_benchmark_task(task, candidate_forward, **kwargs):
        name = kwargs["candidate_name"]
        if "_a000_" in name:
            speedup = 0.5
        else:
            index = int(name.rsplit("c", 1)[1])
            speedup = perf_speedups[min(index, len(perf_speedups) - 1)]
        eager_ms = 10.0
        candidate_ms = eager_ms / speedup
        return BenchmarkResult(
            task_id=task.task_id,
            candidate_name=name,
            shape=(16,),
            dtype="float32",
            device="cpu",
            eager=RuntimeStats(eager_ms, eager_ms, eager_ms, eager_ms, [eager_ms]),
            candidate=RuntimeStats(candidate_ms, candidate_ms, candidate_ms, candidate_ms, [candidate_ms]),
            speedup_vs_eager=speedup,
        )

    monkeypatch.setattr(runner_module, "benchmark_task", fake_benchmark_task)
    config = RunConfig(
        tasks=["vector_add"],
        output_dir=str(tmp_path / "runs"),
        agent=AgentConfig(
            type="llm",
            backend="fake",
            fake_mode="correct",
            max_attempts=1,
            candidates_per_attempt=1,
            stop_after_first_correct=True,
            benchmark_all_correct=True,
            allow_torch_fallback=True,
            performance_search=PerformanceSearchConfig(
                enabled=True,
                max_rounds=perf_max_rounds,
                candidates_per_round=perf_candidates_per_round,
                target_speedup_vs_eager=target,
            ),
        ),
        verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
        benchmark=BenchmarkConfig(enabled=True, device="cpu", max_shapes_per_task=1),
    )
    return run_from_config(config)


def _candidate_records(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("record_type") == "candidate"
    ]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _synthetic_baseline_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "baseline"
    run_dir.mkdir()
    (run_dir / "environment_probe.json").write_text("{}", encoding="utf-8")
    (run_dir / "config.yaml").write_text("agent:\n  backend: fake\n", encoding="utf-8")
    (run_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "record_type": "candidate",
                "task_id": "vector_add",
                "candidate_id": "candidate_000",
                "policy_passed": True,
                "verification_passed": True,
                "benchmark_summary": {"speedup_vs_eager": 0.5},
            }
        )
        + "\n"
        + json.dumps(
            {
                "record_type": "task_summary",
                "task_id": "vector_add",
                "verification": {"passed": True},
                "candidate_records": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir
