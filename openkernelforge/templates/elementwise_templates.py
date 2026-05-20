"""Deterministic Triton templates for simple elementwise tasks."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Iterable

from openkernelforge.agents.base import CandidateSpec


DEFAULT_BLOCK_SIZES = (128, 256, 512, 1024, 2048)
DEFAULT_NUM_WARPS = (4, 8)
DEFAULT_NUM_STAGES = (3,)
DEFAULT_CONTIGUOUS_POLICIES = ("none", "wrapper_contiguous")
DEFAULT_OUTPUT_ALLOCATION_POLICIES = ("torch.empty_like",)
DEFAULT_N_ELEMENTS_MODES = ("runtime",)
DEFAULT_FEATURE_DIM_MODES = ("generic",)


@dataclass(frozen=True)
class TemplateVariant:
    """A deterministic Triton template variant."""

    task_id: str
    block_size: int
    num_warps: int
    num_stages: int = 3
    contiguous_policy: str = "none"
    output_allocation_policy: str = "torch.empty_like"
    n_elements_mode: str = "runtime"
    feature_dim_mode: str = "n/a"
    shape_specialized: bool = False
    template_family: str = "elementwise"

    @property
    def template_id(self) -> str:
        policy = self.contiguous_policy.replace("_", "-")
        alloc = self.output_allocation_policy.replace("torch.", "").replace("_", "-")
        return (
            f"{self.task_id}_bs{self.block_size}_nw{self.num_warps}_"
            f"ns{self.num_stages}_{policy}_{alloc}_n{self.n_elements_mode}_"
            f"f{self.feature_dim_mode}"
        )


def generate_elementwise_templates(
    task_id: str,
    *,
    block_sizes: Iterable[int] = DEFAULT_BLOCK_SIZES,
    num_warps: Iterable[int] = DEFAULT_NUM_WARPS,
    num_stages: Iterable[int] = DEFAULT_NUM_STAGES,
    contiguous_policies: Iterable[str] = DEFAULT_CONTIGUOUS_POLICIES,
    output_allocation_policies: Iterable[str] = DEFAULT_OUTPUT_ALLOCATION_POLICIES,
    n_elements_modes: Iterable[str] = DEFAULT_N_ELEMENTS_MODES,
    feature_dim_modes: Iterable[str] = DEFAULT_FEATURE_DIM_MODES,
) -> list[CandidateSpec]:
    """Generate all deterministic template candidates for a supported task."""

    specs: list[CandidateSpec] = []
    for block_size in block_sizes:
        for warp_count in num_warps:
            for stage_count in num_stages:
                for contiguous_policy in contiguous_policies:
                    for output_policy in output_allocation_policies:
                        for n_mode in n_elements_modes:
                            for feature_mode in _feature_modes_for_task(task_id, feature_dim_modes):
                                variant = TemplateVariant(
                                    task_id=task_id,
                                    block_size=int(block_size),
                                    num_warps=int(warp_count),
                                    num_stages=int(stage_count),
                                    contiguous_policy=str(contiguous_policy),
                                    output_allocation_policy=str(output_policy),
                                    n_elements_mode=str(n_mode),
                                    feature_dim_mode=str(feature_mode),
                                    shape_specialized=(
                                        str(n_mode) == "constexpr" or str(feature_mode) == "constexpr"
                                    ),
                                )
                                specs.append(_candidate_for_variant(variant))
    return specs


def render_vector_add_template(
    *,
    block_size: int,
    num_warps: int,
    num_stages: int = 3,
    contiguous_policy: str = "none",
    output_allocation_policy: str = "torch.empty_like",
    n_elements_mode: str = "runtime",
) -> str:
    contiguous = _contiguous_lines(("x", "y"), contiguous_policy)
    output = _output_allocation_line("x", output_allocation_policy)
    n_param, n_call_arg, n_expr = _n_elements_template_parts(n_elements_mode)
    template = textwrap.dedent(
        f"""
        import torch
        import triton
        import triton.language as tl


        @triton.jit
        def _vector_add_kernel(x_ptr, y_ptr, out_ptr{n_param}, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < {n_expr}
            x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
            tl.store(out_ptr + offsets, x_vals + y_vals, mask=mask)


        def forward(*args):
            x, y = args
            # __CONTIGUOUS_LINES__
            {output}
            n_elements = x.numel()
            grid = (triton.cdiv(n_elements, {block_size}),)
            _vector_add_kernel[grid](x, y, output{n_call_arg}, BLOCK_SIZE={block_size}, num_warps={num_warps}, num_stages={num_stages})
            return output
        """
    )
    return _with_contiguous(template, contiguous)


def render_relu_template(
    *,
    block_size: int,
    num_warps: int,
    num_stages: int = 3,
    contiguous_policy: str = "none",
    output_allocation_policy: str = "torch.empty_like",
    n_elements_mode: str = "runtime",
) -> str:
    contiguous = _contiguous_lines(("x",), contiguous_policy)
    output = _output_allocation_line("x", output_allocation_policy)
    n_param, n_call_arg, n_expr = _n_elements_template_parts(n_elements_mode)
    template = textwrap.dedent(
        f"""
        import torch
        import triton
        import triton.language as tl


        @triton.jit
        def _relu_kernel(x_ptr, out_ptr{n_param}, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < {n_expr}
            vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            result = tl.maximum(vals, 0.0)
            tl.store(out_ptr + offsets, result, mask=mask)


        def forward(*args):
            x, = args
            # __CONTIGUOUS_LINES__
            {output}
            n_elements = x.numel()
            grid = (triton.cdiv(n_elements, {block_size}),)
            _relu_kernel[grid](x, output{n_call_arg}, BLOCK_SIZE={block_size}, num_warps={num_warps}, num_stages={num_stages})
            return output
        """
    )
    return _with_contiguous(template, contiguous)


def render_bias_relu_template(
    *,
    block_size: int,
    num_warps: int,
    num_stages: int = 3,
    contiguous_policy: str = "none",
    output_allocation_policy: str = "torch.empty_like",
    n_elements_mode: str = "runtime",
    feature_dim_mode: str = "generic",
) -> str:
    contiguous = _contiguous_lines(("x", "bias"), contiguous_policy)
    output = _output_allocation_line("x", output_allocation_policy)
    params, call_args, n_expr, f_expr = _bias_relu_template_parts(
        n_elements_mode,
        feature_dim_mode,
    )
    template = textwrap.dedent(
        f"""
        import torch
        import triton
        import triton.language as tl


        @triton.jit
        def _bias_relu_kernel(x_ptr, bias_ptr, out_ptr{params}, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < {n_expr}
            feature_idx = offsets % {f_expr}
            x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            bias_vals = tl.load(bias_ptr + feature_idx, mask=mask, other=0.0)
            result = tl.maximum(x_vals + bias_vals, 0.0)
            tl.store(out_ptr + offsets, result, mask=mask)


        def forward(*args):
            x, bias = args
            # __CONTIGUOUS_LINES__
            {output}
            n_elements = x.numel()
            feature_dim = bias.numel()
            grid = (triton.cdiv(n_elements, {block_size}),)
            _bias_relu_kernel[grid](x, bias, output{call_args}, BLOCK_SIZE={block_size}, num_warps={num_warps}, num_stages={num_stages})
            return output
        """
    )
    return _with_contiguous(template, contiguous)


def _candidate_for_variant(variant: TemplateVariant) -> CandidateSpec:
    renderers = {
        "vector_add": render_vector_add_template,
        "relu": render_relu_template,
        "bias_relu": render_bias_relu_template,
    }
    if variant.task_id not in renderers:
        raise KeyError(f"No elementwise template for task '{variant.task_id}'")
    source = renderers[variant.task_id](
        block_size=variant.block_size,
        num_warps=variant.num_warps,
        num_stages=variant.num_stages,
        contiguous_policy=variant.contiguous_policy,
        output_allocation_policy=variant.output_allocation_policy,
        n_elements_mode=variant.n_elements_mode,
        **({"feature_dim_mode": variant.feature_dim_mode} if variant.task_id == "bias_relu" else {}),
    )
    return CandidateSpec(
        name=f"template_{variant.template_id}",
        source=source,
        metadata={
            "agent": "template",
            "backend": "triton_template",
            "generation_stage": "template_baseline",
            "template_family": variant.template_family,
            "template_id": variant.template_id,
            "block_size": variant.block_size,
            "num_warps": variant.num_warps,
            "num_stages": variant.num_stages,
            "contiguous_policy": variant.contiguous_policy,
            "output_allocation_policy": variant.output_allocation_policy,
            "shape_specialized": variant.shape_specialized,
            "feature_dim_mode": variant.feature_dim_mode,
            "n_elements_mode": variant.n_elements_mode,
        },
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
    if policy == "torch.empty_strided_like_shape":
        return (
            f"output = torch.empty_strided({input_name}.shape, {input_name}.stride(), "
            f"device={input_name}.device, dtype={input_name}.dtype)"
        )
    if policy == "torch.empty":
        return f"output = torch.empty({input_name}.shape, device={input_name}.device, dtype={input_name}.dtype)"
    raise ValueError(f"Unknown output allocation policy: {policy}")


def _n_elements_template_parts(mode: str) -> tuple[str, str, str]:
    if mode == "runtime":
        return ", n_elements", ", n_elements", "n_elements"
    if mode == "constexpr":
        return ", N_ELEMENTS: tl.constexpr", ", N_ELEMENTS=n_elements", "N_ELEMENTS"
    raise ValueError(f"Unknown n_elements mode: {mode}")


def _feature_dim_template_parts(mode: str) -> tuple[str, str, str]:
    if mode in {"generic", "runtime"}:
        return ", feature_dim", ", feature_dim", "feature_dim"
    if mode == "constexpr":
        return ", FEATURE_DIM: tl.constexpr", ", FEATURE_DIM=feature_dim", "FEATURE_DIM"
    raise ValueError(f"Unknown feature_dim mode: {mode}")


def _bias_relu_template_parts(n_mode: str, feature_mode: str) -> tuple[str, str, str, str]:
    params: list[str] = []
    call_args: list[str] = []
    if n_mode == "runtime":
        params.append("n_elements")
        call_args.append("n_elements")
        n_expr = "n_elements"
    elif n_mode == "constexpr":
        n_expr = "N_ELEMENTS"
    else:
        raise ValueError(f"Unknown n_elements mode: {n_mode}")

    if feature_mode in {"generic", "runtime"}:
        params.append("feature_dim")
        call_args.append("feature_dim")
        f_expr = "feature_dim"
    elif feature_mode == "constexpr":
        f_expr = "FEATURE_DIM"
    else:
        raise ValueError(f"Unknown feature_dim mode: {feature_mode}")

    if n_mode == "constexpr":
        params.append("N_ELEMENTS: tl.constexpr")
        call_args.append("N_ELEMENTS=n_elements")
    if feature_mode == "constexpr":
        params.append("FEATURE_DIM: tl.constexpr")
        call_args.append("FEATURE_DIM=feature_dim")
    return ", " + ", ".join(params), ", " + ", ".join(call_args), n_expr, f_expr


def _feature_modes_for_task(task_id: str, feature_dim_modes: Iterable[str]) -> list[str]:
    if task_id == "bias_relu":
        return [str(mode) for mode in feature_dim_modes]
    return ["n/a"]
