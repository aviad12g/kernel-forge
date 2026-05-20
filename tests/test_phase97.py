import json
from pathlib import Path

from openkernelforge.agents.prompt_templates import build_task_prompt
from openkernelforge.agents.repair import build_repair_prompt
from openkernelforge.config import load_config
from openkernelforge.harness.verifier import VerificationResult
from openkernelforge.reports.failure_taxonomy import (
    CORRECT_AND_FAST,
    CORRECT_BUT_SLOW,
    CORRECT_PROMISING_BUT_SLOW,
    classify_candidate_record,
)
from openkernelforge.reports.gpu_debrief import debrief_gpu_run
from openkernelforge.tasks.simple_tasks import get_task
import scripts.compare_gpu_v1_v2 as compare_gpu


def test_debrief_gpu_run_creates_report_from_synthetic_gpu_run(tmp_path):
    run_dir = _synthetic_gpu_run(tmp_path)
    path = debrief_gpu_run(run_dir)
    text = path.read_text(encoding="utf-8")
    assert "GPU Candidate Debrief" in text
    assert "Slow-But-Correct Candidate Analysis" in text
    assert "Triton Compile Error Analysis" in text


def test_v2_vector_add_prompt_includes_skeleton_hints():
    prompt = build_task_prompt(
        get_task("vector_add"),
        allow_torch_fallback=False,
        prompt_version="v2_task_skeletons",
    )
    assert "Flatten all tensors" in prompt
    assert "BLOCK_SIZE" in prompt
    assert "offsets = pid * BLOCK_SIZE" in prompt
    assert "mask = offsets < n_elements" in prompt
    assert "store x + y" in prompt


def test_v2_relu_prompt_includes_skeleton_hints():
    prompt = build_task_prompt(
        get_task("relu"),
        allow_torch_fallback=False,
        prompt_version="v2_task_skeletons",
    )
    assert "tl.maximum" in prompt
    assert "block vectorization" in prompt


def test_v2_bias_relu_prompt_includes_skeleton_hints():
    prompt = build_task_prompt(
        get_task("bias_relu"),
        allow_torch_fallback=False,
        prompt_version="v2_task_skeletons",
    )
    assert "feature_idx = offsets % features" in prompt
    assert "bias is indexed by the last dimension" in prompt


def test_cuda_repair_prompt_includes_slow_correct_feedback():
    verification = VerificationResult(task_id="vector_add", candidate_name="c", passed=True)
    prompt = build_repair_prompt(
        task=get_task("vector_add"),
        original_task_prompt="task",
        previous_candidate="def forward(x, y): return x",
        verification=verification,
        extra_failure="Performance feedback:\n- speedup_vs_eager: 0.5\n- candidate_median_ms: 0.04\n- eager_median_ms: 0.02",
        repair_prompt_version="v3_cuda_repair",
    )
    assert "Correctness passed but performance is slower" in prompt
    assert "speedup_vs_eager" in prompt
    assert "larger BLOCK_SIZE" in prompt
    assert "Return only Python code" in prompt


def test_cuda_repair_prompt_includes_compile_error_guidance():
    verification = VerificationResult(
        task_id="bias_relu",
        candidate_name="c",
        passed=False,
        error="triton compilation traceback",
    )
    prompt = build_repair_prompt(
        task=get_task("bias_relu"),
        original_task_prompt="task",
        previous_candidate="def forward(x, b): return x",
        verification=verification,
        extra_failure="Triton compile traceback: CompilationError",
        repair_prompt_version="v3_cuda_repair",
    )
    assert "Triton compile traceback guidance" in prompt
    assert "@triton.jit" in prompt
    assert "tl.constexpr" in prompt
    assert "Return only Python code" in prompt


def test_prompt_version_is_recorded_in_candidate_records(tmp_path):
    run_dir = _synthetic_gpu_run(tmp_path)
    records = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    candidate = next(record for record in records if record.get("record_type") == "candidate")
    assert candidate["prompt_version"] == "v2_task_skeletons"
    assert candidate["repair_prompt_version"] == "v3_cuda_repair"


def test_v2_gpu_configs_load_successfully():
    config = load_config("configs/gemini_3_1_flash_lite_baseline_3tasks_gpu_v2.yaml")
    assert config.agent.prompt_version == "v2_task_skeletons"
    assert config.agent.repair_prompt_version == "v3_cuda_repair"
    assert not config.agent.stop_after_first_correct


def test_fastsearch_config_loads_successfully():
    config = load_config("configs/gemini_3_1_flash_lite_baseline_3tasks_gpu_v2_fastsearch.yaml")
    assert config.agent.max_attempts == 4
    assert config.agent.candidates_per_attempt == 4
    assert config.agent.temperature == 0.4


def test_compare_gpu_v1_v2_handles_missing_v2(capsys, tmp_path):
    v1 = _synthetic_gpu_run(tmp_path)
    code = compare_gpu.main(["--v1", str(v1)])
    captured = capsys.readouterr()
    assert code == 0
    assert "Missing v2 run" in captured.out
    assert "configs/gemini_3_1_flash_lite_baseline_3tasks_gpu_v2.yaml" in captured.out


def test_correct_speedup_labels_classify_fast_promising_slow():
    fast = classify_candidate_record(
        {"policy_passed": True, "verification_passed": True, "benchmark_summary": {"speedup_vs_eager": 1.1}}
    )
    promising = classify_candidate_record(
        {"policy_passed": True, "verification_passed": True, "benchmark_summary": {"speedup_vs_eager": 0.85}}
    )
    slow = classify_candidate_record(
        {"policy_passed": True, "verification_passed": True, "benchmark_summary": {"speedup_vs_eager": 0.5}}
    )
    assert fast.failure_type == CORRECT_AND_FAST
    assert promising.failure_type == CORRECT_PROMISING_BUT_SLOW
    assert slow.failure_type == CORRECT_BUT_SLOW


def _synthetic_gpu_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "gpu"
    candidate_dir = run_dir / "candidates" / "vector_add"
    log_dir = run_dir / "logs" / "vector_add"
    prompt_dir = run_dir / "prompts" / "vector_add"
    response_dir = run_dir / "responses" / "vector_add"
    for path in [candidate_dir, log_dir, prompt_dir, response_dir]:
        path.mkdir(parents=True, exist_ok=True)

    candidate_path = candidate_dir / "candidate_000.py"
    candidate_path.write_text(
        "import torch\nimport triton\nimport triton.language as tl\n\n"
        "@triton.jit\ndef _kernel(x, y, out, n, BLOCK_SIZE: tl.constexpr):\n"
        "    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offsets < n\n"
        "    tl.store(out + offsets, tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask), mask=mask)\n\n"
        "def forward(x, y):\n"
        "    out = torch.empty_like(x)\n"
        "    _kernel[(triton.cdiv(x.numel(), 1024),)](x, y, out, x.numel(), BLOCK_SIZE=1024)\n"
        "    return out\n",
        encoding="utf-8",
    )
    prompt = prompt_dir / "candidate_000_prompt.txt"
    response = response_dir / "candidate_000_response.txt"
    log = log_dir / "candidate_001.err.txt"
    prompt.write_text("prompt", encoding="utf-8")
    response.write_text("response", encoding="utf-8")
    log.write_text("triton.compiler.errors.CompilationError: bad constexpr", encoding="utf-8")
    (run_dir / "environment_probe.json").write_text(
        json.dumps(
            {
                "viability": "TRITON_EXECUTION_OK",
                "cuda_available": True,
                "triton_available": True,
                "tiny_triton_kernel_passed": True,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(
        "agent:\n  prompt_version: v2_task_skeletons\n  repair_prompt_version: v3_cuda_repair\n",
        encoding="utf-8",
    )
    candidate = {
        "record_type": "candidate",
        "task_id": "vector_add",
        "task_name": "Vector Add",
        "attempt_index": 0,
        "candidate_index": 0,
        "candidate_id": "candidate_000",
        "candidate_name": "c0",
        "backend": "openai_compatible",
        "model": "test",
        "prompt_path": str(prompt),
        "response_path": str(response),
        "candidate_path": str(candidate_path),
        "policy_passed": True,
        "policy_warnings": [],
        "policy_rejection_reason": None,
        "verification_passed": True,
        "verification_summary": {"passed": True},
        "benchmark_summary": {"speedup_vs_eager": 0.5, "candidate_median_ms": 0.04, "eager_median_ms": 0.02},
        "selected_best": True,
        "failure_reason": None,
        "error_log_path": None,
        "prompt_version": "v2_task_skeletons",
        "repair_prompt_version": "v3_cuda_repair",
    }
    failed = dict(candidate)
    failed.update(
        {
            "candidate_id": "candidate_001",
            "candidate_path": str(candidate_path),
            "verification_passed": False,
            "verification_summary": {
                "passed": False,
                "first_error_type": "exception",
                "first_message": "triton compile CompilationError",
            },
            "benchmark_summary": None,
            "selected_best": False,
            "failure_reason": "exception",
            "error_log_path": str(log),
        }
    )
    task = {
        "record_type": "task_summary",
        "task_id": "vector_add",
        "task_name": "Vector Add",
        "agent_type": "llm",
        "backend": "openai_compatible",
        "candidate_id": "candidate_000",
        "candidate_name": "c0",
        "candidate_path": str(candidate_path),
        "verification": {"passed": True},
        "benchmarks": [],
        "attempts": [],
        "candidate_records": [candidate, failed],
    }
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(record) for record in [candidate, failed, task]) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text("summary", encoding="utf-8")
    (run_dir / "analysis.md").write_text("analysis", encoding="utf-8")
    (run_dir / "real_run_review.md").write_text("review", encoding="utf-8")
    return run_dir
