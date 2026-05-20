"""A deterministic dummy agent used to exercise the harness."""

from __future__ import annotations

import textwrap

from openkernelforge.agents.base import CandidateSpec
from openkernelforge.tasks.base import KernelTask
from openkernelforge.utils.gpu import triton_available


class DummyAgent:
    """Return simple torch candidates, with a tiny Triton path when available."""

    def generate(self, task: KernelTask, *, device: str = "auto") -> CandidateSpec:
        use_triton = device.startswith("cuda") and triton_available()
        if use_triton and task.task_id == "vector_add":
            return CandidateSpec(
                name="dummy_triton_vector_add",
                source=_triton_vector_add_candidate(),
                metadata={"agent": "dummy", "backend": "triton"},
            )
        if use_triton and task.task_id == "relu":
            return CandidateSpec(
                name="dummy_triton_relu",
                source=_triton_relu_candidate(),
                metadata={"agent": "dummy", "backend": "triton"},
            )
        return CandidateSpec(
            name=f"dummy_torch_{task.task_id}",
            source=torch_fallback_candidate(task.task_id),
            metadata={"agent": "dummy", "backend": "torch"},
        )


def torch_fallback_candidate(task_id: str) -> str:
    """Return a simple torch candidate for a built-in task."""
    return _torch_candidate(task_id)


def _torch_candidate(task_id: str) -> str:
    bodies = {
        "vector_add": "return x + y",
        "elementwise_mul": "return x * y",
        "relu": "return torch.relu(x)",
        "bias_relu": "return torch.relu(x + bias)",
        "sigmoid_mul": "return torch.sigmoid(x) * y",
        "row_sum": "return torch.sum(x, dim=1)",
        "layernorm_small": (
            "return torch.nn.functional.layer_norm("
            "x, (x.shape[-1],), weight=weight, bias=bias, eps=1e-5)"
        ),
        "matmul_bias": "return x @ weight + bias",
    }
    signatures = {
        "vector_add": "def forward(x, y):",
        "elementwise_mul": "def forward(x, y):",
        "relu": "def forward(x):",
        "bias_relu": "def forward(x, bias):",
        "sigmoid_mul": "def forward(x, y):",
        "row_sum": "def forward(x):",
        "layernorm_small": "def forward(x, weight, bias):",
        "matmul_bias": "def forward(x, weight, bias):",
    }
    if task_id not in bodies:
        body = f"raise NotImplementedError('No dummy candidate for {task_id}')"
        signature = "def forward(*args):"
    else:
        body = bodies[task_id]
        signature = signatures[task_id]
    return textwrap.dedent(
        f"""
        import torch


        {signature}
            {body}
        """
    ).strip()


def _triton_vector_add_candidate() -> str:
    return textwrap.dedent(
        """
        import torch
        import triton
        import triton.language as tl


        @triton.jit
        def _kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + y, mask=mask)


        def forward(x, y):
            out = torch.empty_like(x)
            n = x.numel()
            block = 1024
            grid = (triton.cdiv(n, block),)
            _kernel[grid](x, y, out, n, BLOCK=block)
            return out
        """
    ).strip()


def _triton_relu_candidate() -> str:
    return textwrap.dedent(
        """
        import torch
        import triton
        import triton.language as tl


        @triton.jit
        def _kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            x = tl.load(x_ptr + offsets, mask=mask)
            zero = tl.zeros((BLOCK,), dtype=tl.float32)
            tl.store(out_ptr + offsets, tl.maximum(x, zero), mask=mask)


        def forward(x):
            out = torch.empty_like(x)
            n = x.numel()
            block = 1024
            grid = (triton.cdiv(n, block),)
            _kernel[grid](x, out, n, BLOCK=block)
            return out
        """
    ).strip()
