import json
from pathlib import Path

from openkernelforge.cli import main
from openkernelforge.reports.fused8_curation import (
    curate_fused8_dataset,
    write_fused8_phase11_conclusion,
)


def test_curate_fused8_dataset_command_works_on_synthetic_runs(tmp_path, monkeypatch):
    template, gemini, guided = _synthetic_triplet(tmp_path)
    monkeypatch.chdir(tmp_path)
    code = main(
        [
            "curate-fused8-dataset",
            "--template-run",
            str(template),
            "--gemini-run",
            str(gemini),
            "--template-guided-run",
            str(guided),
            "--out-dir",
            "datasets/fused8_curated_v1",
        ]
    )
    assert code == 0
    assert (tmp_path / "datasets/fused8_curated_v1/manifest.json").exists()
    assert (tmp_path / "runs/fused8_repeatability_comparison.md").exists()
    assert (tmp_path / "runs/fused8_phase11_conclusion.md").exists()


def test_curated_manifest_counts_rows_correctly(tmp_path, monkeypatch):
    template, gemini, guided = _synthetic_triplet(tmp_path)
    monkeypatch.chdir(tmp_path)
    out = curate_fused8_dataset(
        template_run=template,
        gemini_run=gemini,
        template_guided_run=guided,
        out_dir=tmp_path / "dataset",
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest["counts_by_file"]
    for filename, count in counts.items():
        if filename.endswith(".jsonl"):
            assert count == len(_read_jsonl(out / filename))
    assert counts["correct_fast_repeat_stable.jsonl"] >= 1
    assert counts["optimization_pairs_template_vs_gemini.jsonl"] == 1
    assert counts["optimization_pairs_gemini_vs_template.jsonl"] >= 1


def test_repeat_stable_rows_require_repeat_median_at_least_one(tmp_path, monkeypatch):
    template, gemini, guided = _synthetic_triplet(tmp_path)
    monkeypatch.chdir(tmp_path)
    out = curate_fused8_dataset(
        template_run=template,
        gemini_run=gemini,
        template_guided_run=guided,
        out_dir=tmp_path / "dataset",
    )
    stable = _read_jsonl(out / "correct_fast_repeat_stable.jsonl")
    assert stable
    assert all((row["repeatability"]["stats"]["median"] >= 1.0) for row in stable)
    assert all(row["repeatability"].get("stable") for row in stable)


def test_single_run_only_rows_are_separated_from_repeat_stable(tmp_path, monkeypatch):
    template, gemini, guided = _synthetic_triplet(tmp_path)
    monkeypatch.chdir(tmp_path)
    out = curate_fused8_dataset(
        template_run=template,
        gemini_run=gemini,
        template_guided_run=guided,
        out_dir=tmp_path / "dataset",
    )
    single = _read_jsonl(out / "correct_fast_single_run.jsonl")
    stable = _read_jsonl(out / "correct_fast_repeat_stable.jsonl")
    assert any(row["task_id"] == "add_relu" for row in single)
    assert not any(row["task_id"] == "add_relu" for row in stable)


def test_optimization_pair_export_works_both_directions(tmp_path, monkeypatch):
    template, gemini, guided = _synthetic_triplet(tmp_path)
    monkeypatch.chdir(tmp_path)
    out = curate_fused8_dataset(
        template_run=template,
        gemini_run=gemini,
        template_guided_run=guided,
        out_dir=tmp_path / "dataset",
    )
    template_wins = _read_jsonl(out / "optimization_pairs_template_vs_gemini.jsonl")
    gemini_wins = _read_jsonl(out / "optimization_pairs_gemini_vs_template.jsonl")
    assert template_wins[0]["task_id"] == "bias_gelu"
    assert template_wins[0]["fast_source_type"] == "template"
    assert gemini_wins[0]["task_id"] == "bias_relu"
    assert gemini_wins[0]["fast_source_type"] == "gemini"


def test_fused8_conclusion_report_generated_from_synthetic_summaries(tmp_path, monkeypatch):
    template, gemini, guided = _synthetic_triplet(tmp_path)
    monkeypatch.chdir(tmp_path)
    out = curate_fused8_dataset(
        template_run=template,
        gemini_run=gemini,
        template_guided_run=guided,
        out_dir=tmp_path / "dataset",
    )
    report = write_fused8_phase11_conclusion(
        template_run=template,
        gemini_run=gemini,
        template_guided_run=guided,
        dataset_dir=out,
        out=tmp_path / "conclusion.md",
    )
    text = report.read_text(encoding="utf-8")
    assert "Fused8 Conclusion" in text
    assert "not KernelBench" in text
    assert "Curated Dataset Counts" in text


def _synthetic_triplet(tmp_path: Path) -> tuple[Path, Path, Path]:
    template = _make_run(
        tmp_path / "runs/template",
        source="template",
        candidates=[
            ("bias_gelu", 1.5, 1.4),
            ("bias_relu", 1.0, 0.95),
            ("add_relu", 0.85, None),
        ],
    )
    gemini = _make_run(
        tmp_path / "runs/gemini",
        source="gemini",
        candidates=[
            ("bias_gelu", 1.2, 1.1),
            ("bias_relu", 1.3, 1.25),
            ("add_relu", 1.1, None),
            ("sigmoid_mul", 0.9, None),
        ],
    )
    guided = _make_run(
        tmp_path / "runs/guided",
        source="gemini_template_guided",
        candidates=[
            ("bias_gelu", 1.1, 1.05),
            ("bias_relu", 1.05, 0.9),
        ],
    )
    return template, gemini, guided


def _make_run(
    run_dir: Path,
    *,
    source: str,
    candidates: list[tuple[str, float, float | None]],
) -> Path:
    run_dir.mkdir(parents=True)
    records = []
    repeat_rows = []
    for index, (task_id, speedup, repeat_median) in enumerate(candidates):
        candidate_id = f"candidate_{index:03d}"
        candidate_dir = run_dir / "candidates" / task_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_dir / f"{candidate_id}.py"
        candidate_path.write_text("def forward(*args):\n    return args[0]\n", encoding="utf-8")
        record = {
            "record_type": "candidate",
            "task_id": task_id,
            "task_name": task_id,
            "candidate_id": candidate_id,
            "candidate_path": str(candidate_path),
            "policy_passed": True,
            "verification_passed": True,
            "benchmark_summary": {
                "speedup_vs_eager": speedup,
                "speedup_vs_torch_compile": speedup * 0.9,
                "candidate_median_ms": 1.0 / speedup,
                "eager_median_ms": 1.0,
            },
            "generation_stage": "template_fused8_wide" if source == "template" else "initial",
            "task_family": "fused8",
            "backend": "triton_template" if source == "template" else "openai_compatible",
            "model": None if source == "template" else "gemini-3.1-flash-lite",
            "template_id": f"{task_id}_{candidate_id}" if source == "template" else None,
        }
        records.append(record)
        if repeat_median is not None:
            repeat_rows.append(
                {
                    "task_id": task_id,
                    "candidate_id": candidate_id,
                    "candidate_path": str(candidate_path),
                    "original_speedup_vs_eager": speedup,
                    "speedup_values": [repeat_median, repeat_median, repeat_median],
                    "stats": {
                        "mean": repeat_median,
                        "median": repeat_median,
                        "std": 0.0,
                        "min": repeat_median,
                        "max": repeat_median,
                        "coefficient_of_variation": 0.0,
                    },
                    "stable": True,
                    "errors": [],
                }
            )
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(
        "agent:\n"
        f"  type: {'template' if source == 'template' else 'llm'}\n"
        f"  backend: {'triton_template' if source == 'template' else 'openai_compatible'}\n",
        encoding="utf-8",
    )
    (run_dir / "environment_probe.json").write_text("{}", encoding="utf-8")
    (run_dir / "repeatability_results.json").write_text(
        json.dumps({"top_k": 3, "repeats": 5, "results": repeat_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
