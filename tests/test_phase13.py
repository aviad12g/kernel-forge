from pathlib import Path

from openkernelforge.config import load_config
import scripts.check_local_model_server as check_local
import scripts.compare_all_fused8_models as compare_all
import scripts.run_local_model_fused8 as run_local


def test_local_model_configs_load():
    for path in [
        "configs/qwen_fused8_gpu_baseline_cheap.yaml",
        "configs/deepseek_fused8_gpu_baseline_cheap.yaml",
        "configs/nemotron_fused8_gpu_baseline_cheap.yaml",
    ]:
        config = load_config(path)
        assert config.agent.type == "llm"
        assert config.agent.backend == "openai_compatible"
        assert config.agent.base_url == "http://localhost:8000/v1"
        assert config.agent.api_key_env is None
        assert config.agent.max_attempts == 1
        assert config.agent.candidates_per_attempt == 1
        assert config.benchmark.enable_torch_compile is True


def test_local_server_check_exits_clearly_when_unavailable(monkeypatch, capsys):
    def fake_request(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(check_local, "_request_json", fake_request)
    code = check_local.main(["--base-url", "http://localhost:8000/v1", "--model", "test-model"])
    out = capsys.readouterr().out
    assert code == 1
    assert "Local server" in out
    assert "Chat completion health check failed" in out


def test_run_local_model_fused8_exits_when_server_unavailable(monkeypatch, capsys):
    def fake_run_command(args):
        if "env-check" in " ".join(args):
            return _completed(args, stdout="Viability: TRITON_EXECUTION_OK\n")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(run_local, "run_command", fake_run_command)
    monkeypatch.setattr(
        run_local,
        "check_local_model_server",
        lambda **kwargs: check_local.LocalServerCheckResult(
            available=False,
            base_url="http://localhost:8000/v1",
            model="test-model",
            message="connection refused",
        ),
    )
    code = run_local.main(["--config", "configs/qwen_fused8_gpu_baseline_cheap.yaml"])
    out = capsys.readouterr().out
    assert code == 1
    assert "Local model server unavailable" in out
    assert "vllm.entrypoints.openai.api_server" in out


def test_compare_all_fused8_models_handles_missing_runs(tmp_path, capsys):
    missing = tmp_path / "missing"
    code = compare_all.main(
        [
            "--template",
            str(missing / "template"),
            "--gemini",
            str(missing / "gemini"),
            "--gemini-guided",
            str(missing / "gemini-guided"),
            "--openai-mini",
            str(missing / "mini"),
            "--gpt55",
            str(missing / "gpt55"),
            "--qwen",
            str(missing / "qwen"),
            "--out",
            str(tmp_path / "comparison.md"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "No requested run directories were found" in out
    assert "qwen_fused8_gpu_baseline_cheap.yaml" in out
    assert (tmp_path / "comparison.md").exists()


def test_compare_all_fused8_models_writes_available_synthetic_run(tmp_path, capsys):
    run_dir = _synthetic_run(tmp_path / "runs" / "qwen")
    out_path = tmp_path / "comparison.md"
    code = compare_all.main(
        [
            "--template",
            str(tmp_path / "missing-template"),
            "--gemini",
            str(tmp_path / "missing-gemini"),
            "--gemini-guided",
            str(tmp_path / "missing-gemini-guided"),
            "--openai-mini",
            str(tmp_path / "missing-mini"),
            "--gpt55",
            str(tmp_path / "missing-gpt55"),
            "--qwen",
            str(run_dir),
            "--out",
            str(out_path),
        ]
    )
    text = out_path.read_text(encoding="utf-8")
    assert code == 0
    assert "Fused8 All-Model Comparison" in text
    assert "qwen" in text
    assert "1.200" in text
    assert "Missing Runs" in capsys.readouterr().out


def _synthetic_run(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        """
agent:
  type: llm
  backend: openai_compatible
  provider: local_openai_compatible
  model: test-local-model
""",
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        """
{"record_type":"candidate","task_id":"bias_gelu","candidate_path":"runs/qwen/candidates/bias_gelu/candidate_000.py","policy_passed":true,"verification_passed":true,"benchmark_summary":{"speedup_vs_eager":1.2,"speedup_vs_torch_compile":1.1}}
{"record_type":"candidate","task_id":"row_sum","candidate_path":"runs/qwen/candidates/row_sum/candidate_000.py","policy_passed":true,"verification_passed":true,"benchmark_summary":{"speedup_vs_eager":0.7,"speedup_vs_torch_compile":1.0}}
""".lstrip(),
        encoding="utf-8",
    )
    (run_dir / "repeatability_results.json").write_text(
        """
{"results":[{"task_id":"bias_gelu","stable":true,"stats":{"median":1.1}}]}
""".lstrip(),
        encoding="utf-8",
    )
    return run_dir


def _completed(args, returncode=0, stdout="", stderr=""):
    import subprocess

    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)
