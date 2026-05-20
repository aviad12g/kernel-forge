import json
import subprocess

from openkernelforge import cli
from openkernelforge.config import AgentConfig, BenchmarkConfig, RunConfig, VerificationConfig
from openkernelforge.harness.runner import run_from_config
from openkernelforge.reports.failure_taxonomy import (
    ENV_MISSING_CUDA,
    ENV_MISSING_TRITON,
    POLICY_REJECTED_TORCH_FALLBACK,
    classify_candidate_record,
)
from openkernelforge.utils.env_probe import CPU_ONLY, EnvironmentProbeResult, probe_environment
import scripts.run_gpu_baseline_3tasks as gpu_script


def _completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def test_env_probe_works_on_cpu_only_machine_without_failing():
    result = probe_environment()
    assert result.python_version
    assert result.platform
    assert result.viability in {
        "CPU_ONLY",
        "CUDA_NO_TRITON",
        "TRITON_IMPORT_ONLY",
        "TRITON_EXECUTION_OK",
        "UNKNOWN_BROKEN",
    }


def test_env_check_cli_prints_viability(capsys):
    code = cli.main(["env-check"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Viability:" in captured.out


def test_env_check_cli_writes_json(tmp_path):
    out_path = tmp_path / "environment_probe.json"
    code = cli.main(["env-check", "--out", str(out_path)])
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert code == 0
    assert out_path.exists()
    assert "viability" in data


def test_run_metadata_includes_environment_probe(tmp_path):
    run_dir = run_from_config(
        RunConfig(
            tasks=["vector_add"],
            output_dir=str(tmp_path / "runs"),
            agent=AgentConfig(type="dummy", allow_torch_fallback=True),
            verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
            benchmark=BenchmarkConfig(enabled=False, device="cpu"),
        )
    )
    env_path = run_dir / "environment_probe.json"
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert env_path.exists()
    assert "environment_viability" in metadata


def test_failure_taxonomy_maps_missing_triton_to_environment_failure(tmp_path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("import triton\n\ndef forward(x):\n    return x\n", encoding="utf-8")
    record = {
        "policy_passed": True,
        "verification_passed": False,
        "failure_reason": "exception",
        "candidate_path": str(candidate),
        "verification_summary": {"first_message": "ModuleNotFoundError: No module named 'triton'"},
        "environment_probe": {"triton_available": False, "cuda_available": False},
    }
    result = classify_candidate_record(record)
    assert result.failure_type == ENV_MISSING_TRITON


def test_failure_taxonomy_maps_missing_cuda_to_environment_failure(tmp_path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("import triton\n\ndef forward(x):\n    return x\n", encoding="utf-8")
    record = {
        "policy_passed": True,
        "verification_passed": False,
        "failure_reason": "exception",
        "candidate_path": str(candidate),
        "verification_summary": {"first_message": "CUDA driver not found"},
        "environment_probe": {
            "triton_available": True,
            "cuda_available": False,
            "tiny_triton_kernel_passed": False,
        },
    }
    result = classify_candidate_record(record)
    assert result.failure_type == ENV_MISSING_CUDA


def test_failure_taxonomy_keeps_policy_fallback_rejection():
    record = {
        "policy_passed": False,
        "policy_rejection_reason": "obvious_torch_fallback:direct_add",
        "verification_passed": False,
        "environment_probe": {"triton_available": False, "cuda_available": False},
    }
    result = classify_candidate_record(record)
    assert result.failure_type == POLICY_REJECTED_TORCH_FALLBACK


def test_gpu_baseline_script_refuses_when_environment_not_viable(monkeypatch, capsys):
    monkeypatch.setattr(
        gpu_script,
        "probe_environment",
        lambda: EnvironmentProbeResult(
            python_version="test",
            platform="test",
            torch_available=True,
            cuda_available=False,
            triton_available=False,
            viability=CPU_ONLY,
        ),
    )
    code = gpu_script.main(["--config", "configs/gemini_3_1_flash_lite_baseline_3tasks_gpu.yaml"])
    captured = capsys.readouterr()
    assert code == 1
    assert "GPU baseline refused" in captured.out


def test_summary_includes_environment_viability(tmp_path):
    run_dir = run_from_config(
        RunConfig(
            tasks=["vector_add"],
            output_dir=str(tmp_path / "runs"),
            agent=AgentConfig(type="dummy", allow_torch_fallback=True),
            verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
            benchmark=BenchmarkConfig(enabled=False, device="cpu"),
        )
    )
    text = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "Environment viability:" in text
