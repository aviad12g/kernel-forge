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


def test_policy_rejects_assigned_torch_fallback():
    source = """
import torch

def forward(x):
    y = torch.relu(x)
    return y
"""
    result = check_candidate_policy(source, allow_torch_fallback=False)
    assert not result.passed
    assert result.rejection_reason == "obvious_torch_fallback:torch.relu"


def test_policy_does_not_treat_arbitrary_subscript_call_as_triton():
    source = """
def forward(x):
    handlers = [lambda value: value]
    return handlers[0](x)
"""
    result = check_candidate_policy(source, allow_torch_fallback=True)
    assert result.passed
    assert result.uses_triton is False


def test_policy_rejects_unsafe_import_and_requires_real_triton_launch():
    unsafe = """
import os

def forward(x):
    return x
"""
    result = check_candidate_policy(unsafe, allow_torch_fallback=False)
    assert not result.passed
    assert result.rejection_reason == "disallowed_import:os"

    missing = """
import triton
import triton.language as tl

def forward(x):
    return x
"""
    result = check_candidate_policy(
        missing,
        allow_torch_fallback=False,
        require_triton=True,
    )
    assert not result.passed
    assert result.rejection_reason == "missing_triton_kernel_launch"


def test_policy_rejects_from_torch_compute_alias():
    source = """
from torch import relu

def forward(x):
    return relu(x)
"""
    result = check_candidate_policy(source, allow_torch_fallback=False)
    assert not result.passed
    assert result.rejection_reason == "disallowed_from_torch_import:relu"


def test_policy_rejects_import_time_side_effect():
    source = """
import torch
torch.set_default_dtype(torch.float64)

def forward(x):
    return x
"""
    result = check_candidate_policy(source, allow_torch_fallback=False)
    assert not result.passed
    assert result.rejection_reason == "import_time_call:torch.set_default_dtype"


def test_policy_allows_triton_config_at_import_time():
    source = """
import torch
import triton
import triton.language as tl

configs = [triton.Config({'BLOCK': 128}, num_warps=4)]

@triton.jit
def kernel(x, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out + offsets, tl.load(x + offsets, mask=mask), mask=mask)

def forward(x):
    out = torch.empty_like(x)
    kernel[(1,)](x, out, x.numel(), BLOCK=128)
    return out
"""
    result = check_candidate_policy(
        source,
        allow_torch_fallback=False,
        require_triton=True,
    )
    assert result.passed


def test_policy_rejects_import_time_call_in_class_body():
    source = """
import torch

class ModelNew(torch.nn.Module):
    cache = torch.empty(1024)

    def forward(self, x):
        return x
"""

    result = check_candidate_policy(source, allow_torch_fallback=False)

    assert not result.passed
    assert result.rejection_reason == "import_time_call:torch.empty"


def test_policy_rejects_import_time_call_in_function_default():
    source = """
import torch

def forward(x, cache=torch.empty(1024)):
    return x
"""

    result = check_candidate_policy(source, allow_torch_fallback=False)

    assert not result.passed
    assert result.rejection_reason == "import_time_call:torch.empty"


def test_policy_allows_standard_triton_autotune_decorator():
    source = """
import torch
import triton
import triton.language as tl

configs = [triton.Config({'BLOCK': 128}, num_warps=4)]

@triton.autotune(configs=configs, key=['n'])
@triton.jit
def kernel(x, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out + offsets, tl.load(x + offsets, mask=mask), mask=mask)

def forward(x):
    out = torch.empty_like(x)
    kernel[(1,)](x, out, x.numel(), BLOCK=128)
    return out
"""

    result = check_candidate_policy(
        source,
        allow_torch_fallback=False,
        require_triton=True,
    )

    assert result.passed


def test_policy_accepts_modelnew_triton_contract():
    source = """
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x, out, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(out + offsets, tl.load(x + offsets))

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.empty_like(x)
        kernel[(1,)](x, out, x.numel())
        return out
"""
    result = check_candidate_policy(
        source,
        allow_torch_fallback=False,
        require_triton=True,
    )
    assert result.passed
    assert result.uses_triton is True


def test_policy_rejects_torch_fallback_hidden_in_reachable_helper():
    source = """
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x, out, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(out + offsets, tl.load(x + offsets))

def hidden(x):
    return torch.relu(x)

def forward(x):
    return hidden(x)
"""
    result = check_candidate_policy(
        source,
        allow_torch_fallback=False,
        require_triton=True,
    )
    assert not result.passed
    assert result.rejection_reason == "obvious_torch_fallback:torch.relu"


def test_policy_does_not_count_unreachable_triton_launch():
    source = """
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x, out, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(out + offsets, tl.load(x + offsets))

def unused(x):
    out = torch.empty_like(x)
    kernel[(1,)](x, out, x.numel())
    return out

def forward(x):
    return x
"""
    result = check_candidate_policy(
        source,
        allow_torch_fallback=False,
        require_triton=True,
    )
    assert not result.passed
    assert result.rejection_reason == "missing_triton_kernel_launch"


def test_policy_rejects_aliased_torch_import():
    source = """
import torch as t
import triton
import triton.language as tl

@triton.jit
def kernel(x, out, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(out + offsets, tl.load(x + offsets))

def forward(x):
    out = t.empty_like(x)
    kernel[(1,)](x, out, x.numel())
    return out
"""
    result = check_candidate_policy(
        source,
        allow_torch_fallback=False,
        require_triton=True,
    )
    assert not result.passed
    assert result.rejection_reason == "disallowed_import_alias:torch as t"


def test_policy_rejects_locally_bound_torch_compute():
    source = """
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x, out, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(out + offsets, tl.load(x + offsets))

def forward(x):
    hidden = torch.relu
    return hidden(x)
"""
    result = check_candidate_policy(
        source,
        allow_torch_fallback=False,
        require_triton=True,
    )
    assert not result.passed
    assert result.rejection_reason == "obvious_torch_fallback:torch.relu"


def test_policy_rejects_dynamic_attribute_resolution():
    source = """
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x, out, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(out + offsets, tl.load(x + offsets))

def forward(x):
    return getattr(torch, 'relu')(x)
"""
    result = check_candidate_policy(
        source,
        allow_torch_fallback=False,
        require_triton=True,
    )
    assert not result.passed
    assert result.rejection_reason == "unsafe_call:getattr"


def test_policy_rejects_torch_compute_through_direct_input_alias():
    result = check_candidate_policy(
        """
def forward(x):
    hidden = x
    return hidden.square()
""",
        allow_torch_fallback=False,
    )

    assert not result.passed
    assert result.rejection_reason == "obvious_torch_fallback:tensor_method:hidden.square"


def test_policy_rejects_tensor_data_attribute_fallback():
    result = check_candidate_policy(
        """
def forward(x):
    return x.T
""",
        allow_torch_fallback=False,
    )

    assert not result.passed
    assert result.rejection_reason == "obvious_torch_fallback:tensor_attribute:x.T"


def test_policy_allows_tensor_metadata_used_for_triton_wrapper():
    result = check_candidate_policy(
        """
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x, out, n: tl.constexpr):
    pass

def forward(x):
    out = torch.empty_like(x)
    n = x.numel()
    kernel[(1,)](x, out, n)
    return out
""",
        allow_torch_fallback=False,
        require_triton=True,
    )

    assert result.passed


def test_policy_rejects_in_place_torch_compute_on_allocated_output():
    result = check_candidate_policy(
        """
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x, out, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(out + offsets, tl.load(x + offsets))

def forward(x):
    out = torch.empty_like(x)
    kernel[(1,)](x, out, x.numel())
    out.add_(x)
    return out
""",
        allow_torch_fallback=False,
        require_triton=True,
    )

    assert not result.passed
    assert result.rejection_reason == "obvious_torch_fallback:out.add_"


def test_policy_rejects_tensor_copy_fallback():
    result = check_candidate_policy(
        """
import torch

def forward(x):
    out = torch.empty_like(x)
    out.copy_(x)
    return out
""",
        allow_torch_fallback=False,
    )

    assert not result.passed
    assert result.rejection_reason == "obvious_torch_fallback:out.copy_"


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
