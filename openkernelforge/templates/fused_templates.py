"""Deterministic Triton templates for the internal fused8 benchmark."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Iterable

from openkernelforge.agents.base import CandidateSpec


DEFAULT_FUSED_BLOCK_SIZES = (128, 256, 512, 1024, 2048)
DEFAULT_REDUCTION_BLOCK_SIZES = (1024, 2048)
DEFAULT_NUM_WARPS = (1, 2, 4, 8)
DEFAULT_NUM_STAGES = (3, 4)
DEFAULT_CONTIGUOUS_POLICIES = ("none", "wrapper_contiguous")
DEFAULT_OUTPUT_ALLOCATION_POLICIES = ("torch.empty", "torch.empty_like")
DEFAULT_N_ELEMENTS_MODES = ("runtime", "constexpr")
DEFAULT_FEATURE_DIM_MODES = ("runtime", "constexpr")


@dataclass(frozen=True)
class FusedTemplateVariant:
    task_id: str
    block_size: int
    num_warps: int
    num_stages: int
    contiguous_policy: str
    output_allocation_policy: str
    n_elements_mode: str = "runtime"
    feature_dim_mode: str = "runtime"
    reduction_axis: str = "n/a"
    template_family: str = "fused8"

    @property
    def shape_specialized(self) -> bool:
        return self.n_elements_mode == "constexpr" or self.feature_dim_mode == "constexpr"

    @property
    def template_id(self) -> str:
        policy = self.contiguous_policy.replace("_", "-")
        alloc = self.output_allocation_policy.replace("torch.", "").replace("_", "-")
        return (
            f"{self.task_id}_bs{self.block_size}_nw{self.num_warps}_ns{self.num_stages}_"
            f"{policy}_{alloc}_n{self.n_elements_mode}_f{self.feature_dim_mode}"
        )


def generate_fused_templates(
    task_id: str,
    *,
    block_sizes: Iterable[int] = DEFAULT_FUSED_BLOCK_SIZES,
    reduction_block_sizes: Iterable[int] | None = None,
    num_warps: Iterable[int] = DEFAULT_NUM_WARPS,
    num_stages: Iterable[int] = DEFAULT_NUM_STAGES,
    contiguous_policies: Iterable[str] = DEFAULT_CONTIGUOUS_POLICIES,
    output_allocation_policies: Iterable[str] = DEFAULT_OUTPUT_ALLOCATION_POLICIES,
    n_elements_modes: Iterable[str] = DEFAULT_N_ELEMENTS_MODES,
    feature_dim_modes: Iterable[str] = DEFAULT_FEATURE_DIM_MODES,
) -> list[CandidateSpec]:
    """Generate deterministic fused8 template candidates for one task."""

    specs: list[CandidateSpec] = []
    selected_blocks = tuple(reduction_block_sizes or block_sizes)
    for block_size in selected_blocks:
        for warp_count in num_warps:
            for stage_count in num_stages:
                for contiguous_policy in contiguous_policies:
                    for output_policy in _output_policies_for_task(task_id, output_allocation_policies):
                        for n_mode in _n_modes_for_task(task_id, n_elements_modes):
                            for feature_mode in _feature_modes_for_task(task_id, feature_dim_modes):
                                variant = FusedTemplateVariant(
                                    task_id=task_id,
                                    block_size=int(block_size),
                                    num_warps=int(warp_count),
                                    num_stages=int(stage_count),
                                    contiguous_policy=str(contiguous_policy),
                                    output_allocation_policy=str(output_policy),
                                    n_elements_mode=str(n_mode),
                                    feature_dim_mode=str(feature_mode),
                                    reduction_axis="last" if _is_reduction_task(task_id) else "n/a",
                                )
                                specs.append(_candidate_for_variant(variant))
    return specs


def _candidate_for_variant(variant: FusedTemplateVariant) -> CandidateSpec:
    renderer = {
        "bias_relu": _render_bias_relu,
        "sigmoid_mul": _render_sigmoid_mul,
        "add_relu": _render_add_relu,
        "residual_add_relu": _render_residual_add_relu,
        "bias_gelu": _render_bias_gelu,
        "row_sum": _render_row_sum,
        "layernorm_small": _render_layernorm,
        "rmsnorm_small": _render_rmsnorm,
    }.get(variant.task_id)
    if renderer is None:
        raise KeyError(f"No fused template for task '{variant.task_id}'")
    source = renderer(variant)
    return CandidateSpec(
        name=f"template_{variant.template_id}",
        source=source,
        metadata={
            "agent": "template",
            "backend": "triton_template",
            "generation_stage": "template_baseline",
            "template_family": variant.template_family,
            "task_family": "fused8",
            "template_id": variant.template_id,
            "block_size": variant.block_size,
            "reduction_block_size": variant.block_size if _is_reduction_task(variant.task_id) else None,
            "num_warps": variant.num_warps,
            "num_stages": variant.num_stages,
            "contiguous_policy": variant.contiguous_policy,
            "output_allocation_policy": variant.output_allocation_policy,
            "shape_specialized": variant.shape_specialized,
            "feature_dim_mode": variant.feature_dim_mode,
            "n_elements_mode": variant.n_elements_mode,
            "reduction_axis": variant.reduction_axis,
        },
    )


def _render_bias_relu(v: FusedTemplateVariant) -> str:
    return _render_bias_elementwise(
        v,
        kernel_name="_bias_relu_kernel",
        expression="tl.maximum(x_vals + bias_vals, 0.0)",
        args="x, bias = args",
        inputs=("x", "bias"),
    )


def _render_residual_add_relu(v: FusedTemplateVariant) -> str:
    contiguous = _contiguous_lines(("x", "residual", "bias"), v.contiguous_policy)
    output = _output_allocation_line("x", v.output_allocation_policy)
    params, call_args, n_expr, f_expr = _bias_template_parts(v.n_elements_mode, v.feature_dim_mode)
    return _with_contiguous(
        textwrap.dedent(
            f"""
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def _residual_add_relu_kernel(x_ptr, residual_ptr, bias_ptr, out_ptr{params}, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(0)
                offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = offsets < {n_expr}
                feature_idx = offsets % {f_expr}
                x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                residual_vals = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
                bias_vals = tl.load(bias_ptr + feature_idx, mask=mask, other=0.0)
                result = tl.maximum(x_vals + residual_vals + bias_vals, 0.0)
                tl.store(out_ptr + offsets, result, mask=mask)


            def forward(*args):
                x, residual, bias = args
                # __CONTIGUOUS_LINES__
                {output}
                n_elements = x.numel()
                feature_dim = bias.numel()
                grid = (triton.cdiv(n_elements, {v.block_size}),)
                _residual_add_relu_kernel[grid](x, residual, bias, output{call_args}, BLOCK_SIZE={v.block_size}, num_warps={v.num_warps}, num_stages={v.num_stages})
                return output
            """
        ),
        contiguous,
    )


def _render_bias_gelu(v: FusedTemplateVariant) -> str:
    return _render_bias_elementwise(
        v,
        kernel_name="_bias_gelu_kernel",
        expression="shifted * (1.0 / (1.0 + tl.exp(-1.702 * shifted)))",
        prelude="shifted = x_vals + bias_vals",
        args="x, bias = args",
        inputs=("x", "bias"),
    )


def _render_sigmoid_mul(v: FusedTemplateVariant) -> str:
    contiguous = _contiguous_lines(("x", "z"), v.contiguous_policy)
    output = _output_allocation_line("x", v.output_allocation_policy)
    n_param, n_call_arg, n_expr = _n_elements_template_parts(v.n_elements_mode)
    return _with_contiguous(
        textwrap.dedent(
            f"""
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def _sigmoid_mul_kernel(x_ptr, z_ptr, out_ptr{n_param}, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(0)
                offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = offsets < {n_expr}
                x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                z_vals = tl.load(z_ptr + offsets, mask=mask, other=0.0)
                sigmoid = 1.0 / (1.0 + tl.exp(-x_vals))
                tl.store(out_ptr + offsets, sigmoid * z_vals, mask=mask)


            def forward(*args):
                x, z = args
                # __CONTIGUOUS_LINES__
                {output}
                n_elements = x.numel()
                grid = (triton.cdiv(n_elements, {v.block_size}),)
                _sigmoid_mul_kernel[grid](x, z, output{n_call_arg}, BLOCK_SIZE={v.block_size}, num_warps={v.num_warps}, num_stages={v.num_stages})
                return output
            """
        ),
        contiguous,
    )


def _render_add_relu(v: FusedTemplateVariant) -> str:
    contiguous = _contiguous_lines(("x", "y"), v.contiguous_policy)
    output = _output_allocation_line("x", v.output_allocation_policy)
    n_param, n_call_arg, n_expr = _n_elements_template_parts(v.n_elements_mode)
    return _with_contiguous(
        textwrap.dedent(
            f"""
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def _add_relu_kernel(x_ptr, y_ptr, out_ptr{n_param}, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(0)
                offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = offsets < {n_expr}
                x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
                tl.store(out_ptr + offsets, tl.maximum(x_vals + y_vals, 0.0), mask=mask)


            def forward(*args):
                x, y = args
                # __CONTIGUOUS_LINES__
                {output}
                n_elements = x.numel()
                grid = (triton.cdiv(n_elements, {v.block_size}),)
                _add_relu_kernel[grid](x, y, output{n_call_arg}, BLOCK_SIZE={v.block_size}, num_warps={v.num_warps}, num_stages={v.num_stages})
                return output
            """
        ),
        contiguous,
    )


def _render_bias_elementwise(
    v: FusedTemplateVariant,
    *,
    kernel_name: str,
    expression: str,
    args: str,
    inputs: tuple[str, ...],
    prelude: str = "",
) -> str:
    contiguous = _contiguous_lines(inputs, v.contiguous_policy)
    output = _output_allocation_line("x", v.output_allocation_policy)
    params, call_args, n_expr, f_expr = _bias_template_parts(v.n_elements_mode, v.feature_dim_mode)
    prelude_line = f"{prelude}\n                " if prelude else ""
    return _with_contiguous(
        textwrap.dedent(
            f"""
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def {kernel_name}(x_ptr, bias_ptr, out_ptr{params}, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(0)
                offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = offsets < {n_expr}
                feature_idx = offsets % {f_expr}
                x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                bias_vals = tl.load(bias_ptr + feature_idx, mask=mask, other=0.0)
                {prelude_line}result = {expression}
                tl.store(out_ptr + offsets, result, mask=mask)


            def forward(*args):
                {args}
                # __CONTIGUOUS_LINES__
                {output}
                n_elements = x.numel()
                feature_dim = bias.numel()
                grid = (triton.cdiv(n_elements, {v.block_size}),)
                {kernel_name}[grid](x, bias, output{call_args}, BLOCK_SIZE={v.block_size}, num_warps={v.num_warps}, num_stages={v.num_stages})
                return output
            """
        ),
        contiguous,
    )


def _render_row_sum(v: FusedTemplateVariant) -> str:
    contiguous = _contiguous_lines(("x",), v.contiguous_policy)
    return _with_contiguous(
        textwrap.dedent(
            f"""
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def _row_sum_kernel(x_ptr, out_ptr, FEATURE_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
                row = tl.program_id(0)
                offsets = tl.arange(0, BLOCK_SIZE)
                mask = offsets < FEATURE_DIM
                vals = tl.load(x_ptr + row * FEATURE_DIM + offsets, mask=mask, other=0.0)
                total = tl.sum(vals, axis=0)
                tl.store(out_ptr + row, total)


            def forward(*args):
                x, = args
                # __CONTIGUOUS_LINES__
                rows = x.shape[0]
                feature_dim = x.shape[-1]
                output = torch.empty((rows,), device=x.device, dtype=x.dtype)
                _row_sum_kernel[(rows,)](x, output, FEATURE_DIM=feature_dim, BLOCK_SIZE={v.block_size}, num_warps={v.num_warps}, num_stages={v.num_stages})
                return output
            """
        ),
        contiguous,
    )


def _render_layernorm(v: FusedTemplateVariant) -> str:
    contiguous = _contiguous_lines(("x", "weight", "bias"), v.contiguous_policy)
    return _with_contiguous(
        textwrap.dedent(
            f"""
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def _layernorm_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, FEATURE_DIM: tl.constexpr, EPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
                row = tl.program_id(0)
                offsets = tl.arange(0, BLOCK_SIZE)
                mask = offsets < FEATURE_DIM
                vals = tl.load(x_ptr + row * FEATURE_DIM + offsets, mask=mask, other=0.0)
                mean = tl.sum(vals, axis=0) / FEATURE_DIM
                centered = tl.where(mask, vals - mean, 0.0)
                var = tl.sum(centered * centered, axis=0) / FEATURE_DIM
                inv_std = tl.rsqrt(var + EPS)
                weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
                bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
                result = centered * inv_std * weight + bias
                tl.store(out_ptr + row * FEATURE_DIM + offsets, result, mask=mask)


            def forward(*args):
                x, weight, bias = args
                # __CONTIGUOUS_LINES__
                output = torch.empty_like(x)
                rows = x.shape[0]
                feature_dim = x.shape[-1]
                _layernorm_kernel[(rows,)](x, weight, bias, output, FEATURE_DIM=feature_dim, EPS=1.0e-5, BLOCK_SIZE={v.block_size}, num_warps={v.num_warps}, num_stages={v.num_stages})
                return output
            """
        ),
        contiguous,
    )


def _render_rmsnorm(v: FusedTemplateVariant) -> str:
    contiguous = _contiguous_lines(("x", "weight"), v.contiguous_policy)
    return _with_contiguous(
        textwrap.dedent(
            f"""
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def _rmsnorm_kernel(x_ptr, weight_ptr, out_ptr, FEATURE_DIM: tl.constexpr, EPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
                row = tl.program_id(0)
                offsets = tl.arange(0, BLOCK_SIZE)
                mask = offsets < FEATURE_DIM
                vals = tl.load(x_ptr + row * FEATURE_DIM + offsets, mask=mask, other=0.0)
                mean_square = tl.sum(vals * vals, axis=0) / FEATURE_DIM
                inv_rms = tl.rsqrt(mean_square + EPS)
                weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
                result = vals * inv_rms * weight
                tl.store(out_ptr + row * FEATURE_DIM + offsets, result, mask=mask)


            def forward(*args):
                x, weight = args
                # __CONTIGUOUS_LINES__
                output = torch.empty_like(x)
                rows = x.shape[0]
                feature_dim = x.shape[-1]
                _rmsnorm_kernel[(rows,)](x, weight, output, FEATURE_DIM=feature_dim, EPS=1.0e-5, BLOCK_SIZE={v.block_size}, num_warps={v.num_warps}, num_stages={v.num_stages})
                return output
            """
        ),
        contiguous,
    )


def _contiguous_lines(names: tuple[str, ...], policy: str) -> str:
    if policy in {"none", "no_contiguous_call", "assume_contiguous_no_call"}:
        return "    # Inputs are used as provided; benchmark inputs are expected contiguous."
    if policy == "wrapper_contiguous":
        return "\n".join(f"    {name} = {name}.contiguous()" for name in names)
    raise ValueError(f"Unknown contiguous policy: {policy}")


def _with_contiguous(template: str, contiguous: str) -> str:
    return template.replace("    # __CONTIGUOUS_LINES__", contiguous).strip()


def _output_allocation_line(input_name: str, policy: str) -> str:
    if policy == "torch.empty_like":
        return f"output = torch.empty_like({input_name})"
    if policy == "torch.empty":
        return f"output = torch.empty({input_name}.shape, device={input_name}.device, dtype={input_name}.dtype)"
    if policy == "torch.empty_strided_like_shape":
        return (
            f"output = torch.empty_strided({input_name}.shape, {input_name}.stride(), "
            f"device={input_name}.device, dtype={input_name}.dtype)"
        )
    raise ValueError(f"Unknown output allocation policy: {policy}")


def _n_elements_template_parts(mode: str) -> tuple[str, str, str]:
    if mode == "runtime":
        return ", n_elements", ", n_elements", "n_elements"
    if mode == "constexpr":
        return ", N_ELEMENTS: tl.constexpr", ", N_ELEMENTS=n_elements", "N_ELEMENTS"
    raise ValueError(f"Unknown n_elements mode: {mode}")


def _bias_template_parts(n_mode: str, feature_mode: str) -> tuple[str, str, str, str]:
    positional_params: list[str] = []
    constexpr_params: list[str] = []
    positional_call_args: list[str] = []
    keyword_call_args: list[str] = []
    if n_mode == "runtime":
        positional_params.append("n_elements")
        positional_call_args.append("n_elements")
        n_expr = "n_elements"
    elif n_mode == "constexpr":
        constexpr_params.append("N_ELEMENTS: tl.constexpr")
        keyword_call_args.append("N_ELEMENTS=n_elements")
        n_expr = "N_ELEMENTS"
    else:
        raise ValueError(f"Unknown n_elements mode: {n_mode}")

    if feature_mode in {"generic", "runtime"}:
        positional_params.append("feature_dim")
        positional_call_args.append("feature_dim")
        f_expr = "feature_dim"
    elif feature_mode == "constexpr":
        constexpr_params.append("FEATURE_DIM: tl.constexpr")
        keyword_call_args.append("FEATURE_DIM=feature_dim")
        f_expr = "FEATURE_DIM"
    else:
        raise ValueError(f"Unknown feature_dim mode: {feature_mode}")
    params = positional_params + constexpr_params
    call_args = positional_call_args + keyword_call_args
    return ", " + ", ".join(params), ", " + ", ".join(call_args), n_expr, f_expr


def _is_reduction_task(task_id: str) -> bool:
    return task_id in {"row_sum", "layernorm_small", "rmsnorm_small"}


def _feature_modes_for_task(task_id: str, modes: Iterable[str]) -> list[str]:
    if task_id in {"bias_relu", "residual_add_relu", "bias_gelu", "row_sum", "layernorm_small", "rmsnorm_small"}:
        return [str(mode) for mode in modes]
    return ["n/a"]


def _n_modes_for_task(task_id: str, modes: Iterable[str]) -> list[str]:
    if _is_reduction_task(task_id):
        return ["n/a"]
    return [str(mode) for mode in modes]


def _output_policies_for_task(task_id: str, policies: Iterable[str]) -> list[str]:
    if _is_reduction_task(task_id):
        return ["torch.empty"]
    return [str(policy) for policy in policies]
