"""Correctness verifier for candidate kernels."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from openkernelforge.tasks.base import KernelTask, Shape
from openkernelforge.utils.gpu import dtype_from_name, resolve_device, synchronize_if_cuda


@dataclass
class VerificationCaseResult:
    seed: int
    shape: tuple[int, ...]
    passed: bool
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    error_type: str | None = None
    message: str | None = None
    output_shape: list[list[int]] | None = None
    reference_shape: list[list[int]] | None = None
    output_dtype: list[str] | None = None
    reference_dtype: list[str] | None = None


@dataclass
class VerificationResult:
    task_id: str
    candidate_name: str
    passed: bool
    cases: list[VerificationCaseResult] = field(default_factory=list)
    elapsed_s: float = 0.0
    error: str | None = None


def verify_candidate(
    task: KernelTask,
    candidate_forward: Callable[..., Any],
    *,
    candidate_name: str = "candidate",
    seeds: list[int] | None = None,
    shapes: list[Shape] | None = None,
    dtype: str | torch.dtype | None = None,
    device: str | torch.device = "auto",
    rtol: float | None = None,
    atol: float | None = None,
) -> VerificationResult:
    start = time.perf_counter()
    selected_seeds = seeds or [0, 1]
    selected_shapes = shapes or task.benchmark_shapes[:1]
    selected_dtype = dtype_from_name(dtype) if isinstance(dtype, str) else (dtype or task.allowed_dtypes[0])
    selected_device = resolve_device(device)
    selected_rtol = task.tolerance.rtol if rtol is None else rtol
    selected_atol = task.tolerance.atol if atol is None else atol

    cases: list[VerificationCaseResult] = []
    top_error: str | None = None

    for shape in selected_shapes:
        for seed in selected_seeds:
            try:
                inputs = task.generate_inputs(seed, shape, selected_dtype, selected_device)
                reference_inputs = _clone_inputs(inputs)
                candidate_inputs = _clone_inputs(inputs)

                with torch.no_grad():
                    reference_output = task.reference_fn(*reference_inputs)
                    candidate_output = candidate_forward(*candidate_inputs)
                    synchronize_if_cuda(selected_device)

                case = _compare_outputs(
                    seed=seed,
                    shape=shape,
                    candidate_output=candidate_output,
                    reference_output=reference_output,
                    rtol=selected_rtol,
                    atol=selected_atol,
                )
            except Exception:
                tb = traceback.format_exc()
                top_error = tb
                case = VerificationCaseResult(
                    seed=seed,
                    shape=tuple(shape),
                    passed=False,
                    error_type="exception",
                    message=tb,
                )
            cases.append(case)

    elapsed = time.perf_counter() - start
    passed = bool(cases) and all(case.passed for case in cases)
    return VerificationResult(
        task_id=task.task_id,
        candidate_name=candidate_name,
        passed=passed,
        cases=cases,
        elapsed_s=elapsed,
        error=None if passed else top_error,
    )


def _clone_inputs(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    cloned: list[Any] = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            cloned.append(item.clone())
        else:
            cloned.append(item)
    return tuple(cloned)


def _flatten_tensor_output(output: Any) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (tuple, list)):
        tensors: list[torch.Tensor] = []
        for item in output:
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"Output item is not a tensor: {type(item)!r}")
            tensors.append(item)
        return tensors
    raise TypeError(f"Output is not a tensor or tensor sequence: {type(output)!r}")


def _compare_outputs(
    *,
    seed: int,
    shape: Shape,
    candidate_output: Any,
    reference_output: Any,
    rtol: float,
    atol: float,
) -> VerificationCaseResult:
    try:
        candidate_tensors = _flatten_tensor_output(candidate_output)
        reference_tensors = _flatten_tensor_output(reference_output)
    except Exception as exc:
        return VerificationCaseResult(
            seed=seed,
            shape=tuple(shape),
            passed=False,
            error_type="bad_output_type",
            message=str(exc),
        )

    output_shapes = [list(t.shape) for t in candidate_tensors]
    reference_shapes = [list(t.shape) for t in reference_tensors]
    output_dtypes = [str(t.dtype).replace("torch.", "") for t in candidate_tensors]
    reference_dtypes = [str(t.dtype).replace("torch.", "") for t in reference_tensors]

    if len(candidate_tensors) != len(reference_tensors):
        return VerificationCaseResult(
            seed=seed,
            shape=tuple(shape),
            passed=False,
            error_type="wrong_num_outputs",
            message=(
                f"Candidate returned {len(candidate_tensors)} tensors, "
                f"reference returned {len(reference_tensors)}"
            ),
            output_shape=output_shapes,
            reference_shape=reference_shapes,
            output_dtype=output_dtypes,
            reference_dtype=reference_dtypes,
        )

    max_abs = 0.0
    max_rel = 0.0
    for candidate, reference in zip(candidate_tensors, reference_tensors, strict=True):
        if tuple(candidate.shape) != tuple(reference.shape):
            return VerificationCaseResult(
                seed=seed,
                shape=tuple(shape),
                passed=False,
                error_type="wrong_shape",
                message=f"Candidate shape {tuple(candidate.shape)} != reference shape {tuple(reference.shape)}",
                output_shape=output_shapes,
                reference_shape=reference_shapes,
                output_dtype=output_dtypes,
                reference_dtype=reference_dtypes,
            )
        if candidate.dtype != reference.dtype:
            return VerificationCaseResult(
                seed=seed,
                shape=tuple(shape),
                passed=False,
                error_type="wrong_dtype",
                message=f"Candidate dtype {candidate.dtype} != reference dtype {reference.dtype}",
                output_shape=output_shapes,
                reference_shape=reference_shapes,
                output_dtype=output_dtypes,
                reference_dtype=reference_dtypes,
            )
        if not torch.isfinite(candidate).all():
            return VerificationCaseResult(
                seed=seed,
                shape=tuple(shape),
                passed=False,
                error_type="nonfinite_output",
                message="Candidate output contains NaN or Inf",
                output_shape=output_shapes,
                reference_shape=reference_shapes,
                output_dtype=output_dtypes,
                reference_dtype=reference_dtypes,
            )
        if not torch.isfinite(reference).all():
            return VerificationCaseResult(
                seed=seed,
                shape=tuple(shape),
                passed=False,
                error_type="nonfinite_reference",
                message="Reference output contains NaN or Inf",
                output_shape=output_shapes,
                reference_shape=reference_shapes,
                output_dtype=output_dtypes,
                reference_dtype=reference_dtypes,
            )

        diff = torch.abs(candidate - reference)
        abs_error = float(diff.max().item()) if diff.numel() else 0.0
        denominator = torch.clamp(torch.abs(reference), min=1e-12)
        rel_error = float((diff / denominator).max().item()) if diff.numel() else 0.0
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)

        if not torch.allclose(candidate, reference, rtol=rtol, atol=atol):
            return VerificationCaseResult(
                seed=seed,
                shape=tuple(shape),
                passed=False,
                max_abs_error=max_abs,
                max_rel_error=max_rel,
                error_type="values_not_close",
                message=f"torch.allclose failed with rtol={rtol}, atol={atol}",
                output_shape=output_shapes,
                reference_shape=reference_shapes,
                output_dtype=output_dtypes,
                reference_dtype=reference_dtypes,
            )

    return VerificationCaseResult(
        seed=seed,
        shape=tuple(shape),
        passed=True,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        output_shape=output_shapes,
        reference_shape=reference_shapes,
        output_dtype=output_dtypes,
        reference_dtype=reference_dtypes,
    )
