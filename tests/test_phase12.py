import json
from pathlib import Path

from openkernelforge.agents.backends import OpenAIResponsesBackend
from openkernelforge.cli import main
from openkernelforge.config import RunConfig, load_config
from openkernelforge.reports.fused8_curation import (
    inspect_curated_fused8_dataset,
    validate_curated_fused8_dataset,
)
import scripts.compare_models_fused8 as compare_models
import scripts.run_openai_fused8 as run_openai
import scripts.run_strong_model_fused8 as run_strong


def test_inspect_curated_fused8_works_on_synthetic_dataset(tmp_path):
    dataset = _curated_dataset(tmp_path / "dataset")
    report = inspect_curated_fused8_dataset(dataset)
    text = report.read_text(encoding="utf-8")
    assert "Curated Fused8 Dataset Inspection" in text
    assert "Stable-Fast Rows" in text
    assert "Optimization Pairs" in text


def test_validate_curated_fused8_catches_stable_fast_below_one(tmp_path):
    dataset = _curated_dataset(tmp_path / "dataset")
    rows = _read_jsonl(dataset / "correct_fast_repeat_stable.jsonl")
    rows[0]["repeatability"]["stats"]["median"] = 0.99
    _write_jsonl(dataset / "correct_fast_repeat_stable.jsonl", rows)
    ok, report, errors = validate_curated_fused8_dataset(dataset)
    assert not ok
    assert errors
    assert "below repeat median" in report.read_text(encoding="utf-8")


def test_validate_curated_fused8_passes_valid_dataset(tmp_path):
    dataset = _curated_dataset(tmp_path / "dataset")
    ok, report, errors = validate_curated_fused8_dataset(dataset)
    assert ok
    assert errors == []
    assert "PASS" in report.read_text(encoding="utf-8")


def test_strong_and_openai_configs_load():
    for path in [
        "configs/qwen_fused8_gpu_baseline.yaml",
        "configs/qwen_fused8_gpu_template_guided.yaml",
        "configs/strong_model_fused8_gpu_baseline.yaml",
        "configs/strong_model_fused8_gpu_template_guided.yaml",
        "configs/openai_fused8_gpu_baseline.yaml",
        "configs/openai_fused8_gpu_template_guided.yaml",
        "configs/openai_mini_fused8_gpu_baseline.yaml",
        "configs/openai_mini_fused8_gpu_template_guided.yaml",
        "configs/openai_responses_fused8_gpu_baseline.yaml",
        "configs/openai_responses_fused8_gpu_template_guided.yaml",
    ]:
        config = load_config(path)
        assert config.tasks == [
            "bias_relu",
            "sigmoid_mul",
            "add_relu",
            "residual_add_relu",
            "bias_gelu",
            "row_sum",
            "layernorm_small",
            "rmsnorm_small",
        ]
        assert config.agent.type == "llm"


def test_openai_api_key_redacted_from_safe_config():
    config = RunConfig.from_dict(
        {
            "agent": {
                "type": "llm",
                "backend": "openai_compatible",
                "api_key": "sk-secret",
                "extra_headers": {"Authorization": "Bearer sk-secret"},
            }
        }
    )
    safe = config.to_safe_dict()
    assert safe["agent"]["api_key"] == "<redacted>"
    assert safe["agent"]["extra_headers"]["Authorization"] == "<redacted>"
    assert "sk-secret" not in json.dumps(safe)


def test_run_strong_model_exits_cleanly_when_backend_unavailable(monkeypatch, capsys):
    def fake_run(args):
        joined = " ".join(args)
        if "env-check" in joined:
            return _completed(args, stdout="Viability: TRITON_EXECUTION_OK\n")
        if "check-backend" in joined:
            return _completed(args, returncode=1, stdout="Backend check failed: connection refused\n")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(run_strong, "run_command", fake_run)
    code = run_strong.main(["--baseline-config", "a.yaml", "--guided-config", "b.yaml"])
    assert code == 1
    assert "Backend unavailable" in capsys.readouterr().out


def test_run_openai_fused8_exits_cleanly_when_key_missing(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = run_openai.main([])
    assert code == 1
    out = capsys.readouterr().out
    assert "OPENAI_API_KEY is not set" in out
    assert "export OPENAI_API_KEY=<your-key>" in out


def test_openai_responses_backend_parses_mock_success(monkeypatch):
    class Response:
        status_code = 200
        text = "{}"

        def json(self):
            return {"output": [{"content": [{"type": "output_text", "text": "OK code"}]}]}

    import requests

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Response())
    backend = OpenAIResponsesBackend(model="test-model", api_key="test-key")
    assert backend.generate("hello") == "OK code"


def test_openai_responses_backend_handles_mock_error(monkeypatch):
    class Response:
        status_code = 401
        text = '{"error":"bad auth"}'

        def json(self):
            return {}

    import requests

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Response())
    backend = OpenAIResponsesBackend(model="test-model", api_key="test-key")
    try:
        backend.generate("hello")
    except RuntimeError as exc:
        assert "HTTP 401" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_compare_models_fused8_handles_missing_strong_runs(capsys):
    code = compare_models.main(["--template", "missing-template"])
    assert code == 0
    assert "Missing fused8 model runs" in capsys.readouterr().out


def _curated_dataset(root: Path) -> Path:
    root.mkdir(parents=True)
    rows = {
        "correct_fast_repeat_stable.jsonl": [
            _candidate_row("bias_gelu", "template", 1.5, 1.4, "stable_fast")
        ],
        "correct_fast_single_run.jsonl": [
            _candidate_row("bias_relu", "gemini", 1.1, None, "single_run_fast")
        ],
        "correct_promising.jsonl": [
            _candidate_row("add_relu", "gemini", 0.9, None, "promising")
        ],
        "optimization_pairs_template_vs_gemini.jsonl": [
            _pair_row("bias_gelu", "gemini", "template")
        ],
        "optimization_pairs_gemini_vs_template.jsonl": [
            _pair_row("bias_relu", "template", "gemini")
        ],
        "rejected_or_unstable.jsonl": [
            _candidate_row("row_sum", "template", 0.6, None, "unstable")
        ],
    }
    counts = {}
    for filename, file_rows in rows.items():
        _write_jsonl(root / filename, file_rows)
        counts[filename] = len(file_rows)
    (root / "manifest.json").write_text(
        json.dumps({"counts_by_file": counts, "source_runs": {}}, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def _candidate_row(task_id: str, source: str, speedup: float, repeat: float | None, label: str) -> dict:
    code = "import triton\n\n@triton.jit\ndef kernel():\n    pass\n\ndef forward(*args):\n    return args[0]\n"
    return {
        "task_id": task_id,
        "task_family": "fused8",
        "source_type": source,
        "generation_stage": "initial",
        "candidate_code": code,
        "candidate_path": f"runs/x/candidates/{task_id}/candidate_000.py",
        "benchmark": {"speedup_vs_eager": speedup},
        "repeatability": (
            {
                "stable": True,
                "stats": {
                    "median": repeat,
                    "mean": repeat,
                    "std": 0.0,
                    "coefficient_of_variation": 0.0,
                },
            }
            if repeat is not None
            else {}
        ),
        "template_metadata": {},
        "label": label,
        "speedup_vs_eager": speedup,
        "policy_passed": True,
        "verification_passed": True,
    }


def _pair_row(task_id: str, slow: str, fast: str) -> dict:
    code = "def forward(*args):\n    return args[0]\n"
    return {
        "task_id": task_id,
        "task_family": "fused8",
        "target_type": f"{fast}_wins",
        "label": "optimization_pair",
        "source_type": "template_gemini_comparison",
        "slow_source_type": slow,
        "fast_source_type": fast,
        "slow_code": code,
        "fast_code": code + "# fast\n",
        "target": code + "# fast\n",
        "speedup_delta": 0.2,
        "fast_repeatability": {"stable": True, "stats": {"median": 1.2}},
        "fast_candidate_path": "runs/x/candidates/task/candidate_001.py",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _completed(args, returncode=0, stdout="", stderr=""):
    import subprocess

    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)
