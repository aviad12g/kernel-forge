import json

from openkernelforge.config import AgentConfig, BenchmarkConfig, RunConfig, VerificationConfig
from openkernelforge.harness.policy import check_candidate_policy
from openkernelforge.harness.runner import run_from_config


TRITON_VECTOR_ADD = """
import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def forward(x, y):
    out = torch.empty_like(x)
    grid = (1,)
    _kernel[grid](x, y, out, x.numel(), BLOCK=1024)
    return out
"""


TORCH_FALLBACK = """
import torch


def forward(x, y):
    return x + y
"""


TRITON_WITH_IMPORTERROR_FALLBACK = """
import torch


def forward(x, y):
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
            offsets = tl.arange(0, BLOCK)
            mask = offsets < n
            x_val = tl.load(x_ptr + offsets, mask=mask)
            y_val = tl.load(y_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x_val + y_val, mask=mask)

        out = torch.empty_like(x)
        _kernel[(1,)](x, y, out, x.numel(), BLOCK=1024)
        return out
    except ImportError:
        return x + y
"""


def test_policy_passes_plausible_triton_candidate():
    result = check_candidate_policy(TRITON_VECTOR_ADD, allow_torch_fallback=False)
    assert result.passed
    assert result.has_forward
    assert result.uses_triton


def test_policy_rejects_obvious_torch_fallback_when_disabled():
    result = check_candidate_policy(TORCH_FALLBACK, allow_torch_fallback=False)
    assert not result.passed
    assert result.rejection_reason == "obvious_torch_fallback:direct_add"


def test_policy_rejects_importerror_torch_fallback_when_disabled():
    result = check_candidate_policy(TRITON_WITH_IMPORTERROR_FALLBACK, allow_torch_fallback=False)
    assert not result.passed
    assert result.uses_triton
    assert result.rejection_reason == "obvious_torch_fallback:direct_add"


def test_policy_allows_torch_fallback_when_enabled():
    result = check_candidate_policy(TORCH_FALLBACK, allow_torch_fallback=True)
    assert result.passed
    assert "torch_fallback_allowed" in result.warnings


def test_runner_skips_verifier_and_benchmark_when_policy_rejects(tmp_path):
    config = RunConfig(
        tasks=["vector_add"],
        output_dir=str(tmp_path),
        agent=AgentConfig(
            type="llm",
            backend="fake",
            fake_mode="correct",
            max_attempts=1,
            candidates_per_attempt=1,
            allow_torch_fallback=False,
        ),
        verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
        benchmark=BenchmarkConfig(enabled=True, warmup=1, repeats=2, device="cpu"),
    )
    run_dir = run_from_config(config)
    records = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    candidate = [record for record in records if record.get("record_type") == "candidate"][0]
    assert not candidate["policy_passed"]
    assert candidate["policy_rejection_reason"] == "obvious_torch_fallback:direct_add"
    assert not candidate["verification_passed"]
    assert candidate["benchmark_summary"] is None


def test_results_jsonl_includes_policy_fields(tmp_path):
    config = RunConfig(
        tasks=["vector_add"],
        output_dir=str(tmp_path),
        agent=AgentConfig(type="dummy", allow_torch_fallback=True),
        verification=VerificationConfig(seeds=[0], device="cpu", max_shapes_per_task=1),
        benchmark=BenchmarkConfig(enabled=False, device="cpu"),
    )
    run_dir = run_from_config(config)
    records = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    candidate = [record for record in records if record.get("record_type") == "candidate"][0]
    assert "policy_passed" in candidate
    assert "policy_warnings" in candidate
    assert "policy_rejection_reason" in candidate
