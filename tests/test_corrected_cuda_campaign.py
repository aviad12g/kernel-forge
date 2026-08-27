from pathlib import Path

import pytest

from scripts.run_corrected_cuda_campaign import (
    CampaignError,
    _report_path_from_output,
    _validate_baseline_only_config,
    _validate_stage_data,
)
from scripts.package_corrected_cuda_bundle import package_bundle


def _stage_data(*, task_count: int = 5) -> dict:
    return {
        "status": "completed",
        "tasks_selected": task_count,
        "candidate_records": [],
        "failures": [],
        "records": [
            {
                "task_id": f"task_{index}",
                "candidate_contract": "model_new",
                "reference_ok": True,
                "skipped": False,
                "benchmark_summary": {
                    "benchmark_error": None,
                    "compile_error": None,
                    "eager_median_ms": 1.0,
                    "torch_compile_median_ms": 0.8,
                    "cache_flush_performed": True,
                    "timing_mode": "cuda_event",
                    "independent_sessions": 3,
                    "repeat": 120,
                },
            }
            for index in range(task_count)
        ],
    }


def test_corrected_campaign_configs_are_baseline_only():
    _validate_baseline_only_config(
        Path("configs/kernelbench_l1_5task_corrected_rigorous.yaml"),
        expected_tasks=5,
    )
    _validate_baseline_only_config(
        Path("configs/kernelbench_l1_20task_corrected_rigorous_safe.yaml"),
        expected_tasks=20,
    )


def test_corrected_campaign_config_rejects_candidate_provider(tmp_path):
    config = tmp_path / "unsafe.yaml"
    config.write_text(
        """
kernelbench:
  max_tasks: 5
  candidate_provider: gemini
benchmark:
  timing_mode: cuda_event
  repeat: 120
  independent_sessions: 3
  include_torch_compile: true
  torch_compile_mode: max-autotune
  cache_flush: {enabled: true}
execution:
  require_cuda: true
  require_triton: true
  require_tiny_triton_kernel: true
""",
        encoding="utf-8",
    )

    with pytest.raises(CampaignError, match="candidate_provider must be none"):
        _validate_baseline_only_config(config, expected_tasks=5)


def test_corrected_stage_gate_accepts_complete_modelnew_baseline():
    validation = _validate_stage_data(_stage_data(), expected_tasks=5)

    assert validation["passed"] is True
    assert validation["eager_timed"] == 5
    assert validation["compile_timed"] == 5
    assert validation["cache_perturbed"] == 5


def test_corrected_stage_gate_rejects_historical_contract_and_candidates():
    data = _stage_data()
    data["records"][0]["candidate_contract"] = "forward"
    data["candidate_records"] = [{"task_id": "unexpected"}]

    validation = _validate_stage_data(data, expected_tasks=5)

    assert validation["passed"] is False
    assert any("candidate records" in error for error in validation["errors"])
    assert any("candidate_contract is not model_new" in error for error in validation["errors"])


def test_corrected_stage_gate_rejects_missing_compiler_baseline():
    data = _stage_data()
    data["records"][2]["benchmark_summary"]["torch_compile_median_ms"] = None

    validation = _validate_stage_data(data, expected_tasks=5)

    assert validation["passed"] is False
    assert "torch.compile timing count=4, expected 5" in validation["errors"]


def test_report_path_parser_handles_workspace_spaces(tmp_path):
    output = "KernelBench L1 check written: runs/corrected_kernelbench/abc/kernelbench_l1_check.md\n"

    path = _report_path_from_output(output, tmp_path)

    assert path == (tmp_path / "runs/corrected_kernelbench/abc/kernelbench_l1_check.md").resolve()


def test_cuda_bundle_contains_manifest_and_checksums(tmp_path):
    source = tmp_path / "source"
    (source / "openkernelforge").mkdir(parents=True)
    (source / "openkernelforge" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "pyproject.toml").write_text("[project]\nname = 'okf'\n", encoding="utf-8")
    output = tmp_path / "bundle.tar.gz"

    package_bundle(source, output)

    assert output.is_file()
    import tarfile

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert "bundle_manifest.json" in names
    assert "SHA256SUMS" in names
    assert "openkernelforge/module.py" in names


def test_cuda_bundle_rejects_secret_like_values(tmp_path):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    secret = "rpa" + "_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
    (source / "scripts" / "bad.py").write_text(
        f"TOKEN = {secret!r}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="possible secret"):
        package_bundle(source, tmp_path / "bundle.tar.gz")
