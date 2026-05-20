import subprocess

from openkernelforge import cli
from openkernelforge.agents.prompt_templates import build_task_prompt
from openkernelforge.config import AgentConfig, BenchmarkConfig, RunConfig, VerificationConfig
from openkernelforge.harness.runner import run_from_config
from openkernelforge.tasks.simple_tasks import get_task
import scripts.run_baseline_comparison_3tasks as comparison_script
import scripts.run_real_baseline_3tasks as real_script


def _completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def test_real_baseline_runner_exits_clearly_when_backend_unavailable(monkeypatch, capsys):
    def fake_run_command(args):
        return _completed(args, returncode=1, stdout="Backend check failed: unavailable\n")

    monkeypatch.setattr(real_script, "run_command", fake_run_command)
    code = real_script.main()
    captured = capsys.readouterr()
    assert code == 1
    assert "Backend unavailable; real baseline was not run." in captured.out
    assert "Start an OpenAI-compatible server" in captured.out


def test_real_baseline_runner_with_mocked_available_backend_calls_expected_commands(monkeypatch, capsys):
    calls = []

    def fake_run_command(args):
        calls.append(args)
        joined = " ".join(args)
        if "check-backend" in joined:
            return _completed(args, stdout="Backend check succeeded.\n")
        if " run " in f" {joined} ":
            return _completed(args, stdout="Run complete: runs/mock_real\n")
        if "analyze-run" in joined:
            return _completed(args, stdout="Analysis written: runs/mock_real/analysis.md\n")
        if "export-dataset" in joined:
            return _completed(args, stdout="Dataset exported: datasets/mock\n")
        if "validate-dataset" in joined:
            return _completed(args, stdout="Dataset validation passed.\n")
        return _completed(args)

    monkeypatch.setattr(real_script, "run_command", fake_run_command)
    code = real_script.main()
    captured = capsys.readouterr()
    assert code == 0
    assert any("check-backend" in " ".join(call) for call in calls)
    assert any("analyze-run" in " ".join(call) for call in calls)
    assert any("export-dataset" in " ".join(call) for call in calls)
    assert any("validate-dataset" in " ".join(call) for call in calls)
    assert "Run directory: runs/mock_real" in captured.out


def test_prompt_hardening_includes_no_fallback_instruction_and_task_hints():
    vector_prompt = build_task_prompt(get_task("vector_add"), allow_torch_fallback=False)
    assert "do not use plain PyTorch fallback code" in vector_prompt
    assert "Use Triton when CUDA is available" in vector_prompt
    assert "one output element per input element" in vector_prompt

    bias_prompt = build_task_prompt(get_task("bias_relu"), allow_torch_fallback=False)
    assert "broadcast bias over the last dimension" in bias_prompt


def test_review_real_run_creates_report_and_marks_fake_as_harness_only(tmp_path):
    run_dir = run_from_config(
        RunConfig(
            tasks=["vector_add"],
            output_dir=str(tmp_path / "runs"),
            agent=AgentConfig(type="llm", backend="fake", fake_mode="correct"),
            verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
            benchmark=BenchmarkConfig(enabled=False, device="cpu"),
        )
    )
    code = cli.main(["review-real-run", "--run-dir", str(run_dir)])
    review_path = run_dir / "real_run_review.md"
    text = review_path.read_text(encoding="utf-8")
    assert code == 0
    assert review_path.exists()
    assert "Appears real model run: no (harness-only)" in text


def test_baseline_comparison_script_works_without_real_backend(monkeypatch, capsys):
    calls = []

    def fake_run_command(args):
        calls.append(args)
        joined = " ".join(args)
        if "dummy_baseline_3tasks.yaml" in joined:
            return _completed(args, stdout="Run complete: runs/dummy\n")
        if "fake_baseline_3tasks.yaml" in joined:
            return _completed(args, stdout="Run complete: runs/fake\n")
        if "check-backend" in joined:
            return _completed(args, returncode=1, stdout="Backend check failed: unavailable\n")
        if "scripts/compare_runs.py" in joined:
            return _completed(args, stdout="| Run dir | Agent/backend/model |\n")
        return _completed(args)

    monkeypatch.setattr(comparison_script, "run_command", fake_run_command)
    code = comparison_script.main()
    captured = capsys.readouterr()
    assert code == 0
    assert "Real backend unavailable; skipped real baseline." in captured.out
    assert "| Run dir | Agent/backend/model |" in captured.out
    assert not any("real_baseline_3tasks.yaml" in " ".join(call) and " run " in f" {' '.join(call)} " for call in calls)
