"""Correctness verifier for candidate kernels."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from openkernelforge.harness.inputs import clone_inputs, find_value_difference
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
    output_tree: str | None = None
    reference_tree: str | None = None
    deterministic_repeats: int = 1
    alias_contract_passed: bool | None = None


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
    deterministic_repeats: int = 1,
    require_alias_contract: bool = False,
) -> VerificationResult:
    start = time.perf_counter()
    selected_seeds = seeds or [0, 1]
    selected_shapes = shapes or task.benchmark_shapes[:1]
    selected_dtype = dtype_from_name(dtype) if isinstance(dtype, str) else (dtype or task.allowed_dtypes[0])
    selected_device = resolve_device(device)
    selected_rtol = task.tolerance.rtol if rtol is None else rtol
    selected_atol = task.tolerance.atol if atol is None else atol
    if deterministic_repeats <= 0:
        raise ValueError("deterministic_repeats must be positive")

    reference_callable = _prepare_callable(task.reference_fn, selected_dtype, selected_device)
    candidate_callable = _prepare_callable(candidate_forward, selected_dtype, selected_device)

    cases: list[VerificationCaseResult] = []
    top_error: str | None = None

    for shape in selected_shapes:
        for seed in selected_seeds:
            try:
                inputs = task.generate_inputs(seed, shape, selected_dtype, selected_device)
                reference_inputs = clone_inputs(inputs)
                candidate_inputs = clone_inputs(inputs)
                reference_inputs_before = clone_inputs(reference_inputs)
                candidate_inputs_before = clone_inputs(candidate_inputs)

                with torch.no_grad():
                    reference_output = reference_callable(*reference_inputs)
                    candidate_output = candidate_callable(*candidate_inputs)
                    repeat_outputs = []
                    for _ in range(deterministic_repeats - 1):
                        repeated_inputs = clone_inputs(inputs)
                        repeat_outputs.append(candidate_callable(*repeated_inputs))
                    synchronize_if_cuda(selected_device)

                input_effect_error = _compare_input_effects(
                    seed=seed,
                    shape=shape,
                    reference_before=reference_inputs_before,
                    reference_after=reference_inputs,
                    candidate_before=candidate_inputs_before,
                    candidate_after=candidate_inputs,
                )
                determinism_error = _compare_repeat_outputs(
                    seed=seed,
                    shape=shape,
                    first_output=candidate_output,
                    repeat_outputs=repeat_outputs,
                    deterministic_repeats=deterministic_repeats,
                )
                case = (
                    input_effect_error
                    or determinism_error
                    or _compare_outputs(
                        seed=seed,
                        shape=shape,
                        candidate_output=candidate_output,
                        reference_output=reference_output,
                        candidate_inputs=candidate_inputs,
                        reference_inputs=reference_inputs,
                        require_alias_contract=require_alias_contract,
                        deterministic_repeats=deterministic_repeats,
                        rtol=selected_rtol,
                        atol=selected_atol,
                    )
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


def _compare_input_effects(
    *,
    seed: int,
    shape: Shape,
    reference_before: tuple[Any, ...],
    reference_after: tuple[Any, ...],
    candidate_before: tuple[Any, ...],
    candidate_after: tuple[Any, ...],
) -> VerificationCaseResult | None:
    reference_mutation = find_value_difference(reference_before, reference_after)
    candidate_mutation = find_value_difference(candidate_before, candidate_after)
    if reference_mutation is None and candidate_mutation is not None:
        return VerificationCaseResult(
            seed=seed,
            shape=tuple(shape),
            passed=False,
            error_type="unexpected_input_mutation",
            message=f"Candidate modified {candidate_mutation}, but the reference left inputs unchanged",
        )
    if reference_mutation is not None:
        final_difference = find_value_difference(reference_after, candidate_after)
        if final_difference is not None:
            return VerificationCaseResult(
                seed=seed,
                shape=tuple(shape),
                passed=False,
                error_type="input_mutation_mismatch",
                message=(
                    f"Reference mutates inputs beginning at {reference_mutation}, but candidate "
                    f"input effects differ at {final_difference}"
                ),
            )
    return None


def _prepare_callable(
    fn: Callable[..., Any],
    dtype: torch.dtype,
    device: torch.device,
) -> Callable[..., Any]:
    prepare = getattr(fn, "prepare_for", None)
    if callable(prepare):
        prepared = prepare(dtype, device)
        if callable(prepared):
            return prepared
    return fn


def _flatten_tensor_output(output: Any) -> tuple[tuple[Any, ...], list[torch.Tensor]]:
    if isinstance(output, torch.Tensor):
        return ("tensor",), [output]
    if isinstance(output, tuple):
        tensors: list[torch.Tensor] = []
        children: list[tuple[Any, ...]] = []
        for item in output:
            spec, child_tensors = _flatten_tensor_output(item)
            children.append(spec)
            tensors.extend(child_tensors)
        return ("tuple", tuple(children)), tensors
    if isinstance(output, list):
        tensors = []
        children = []
        for item in output:
            spec, child_tensors = _flatten_tensor_output(item)
            children.append(spec)
            tensors.extend(child_tensors)
        return ("list", tuple(children)), tensors
    if isinstance(output, dict):
        tensors = []
        children = []
        for key, item in output.items():
            spec, child_tensors = _flatten_tensor_output(item)
            children.append((repr(key), spec))
            tensors.extend(child_tensors)
        return ("dict", tuple(children)), tensors
    raise TypeError(f"Output leaf is not a tensor: {type(output)!r}")


def _compare_repeat_outputs(
    *,
    seed: int,
    shape: Shape,
    first_output: Any,
    repeat_outputs: list[Any],
    deterministic_repeats: int,
) -> VerificationCaseResult | None:
    try:
        first_tree, first_tensors = _flatten_tensor_output(first_output)
        for repeat_index, repeated in enumerate(repeat_outputs, start=2):
            repeat_tree, repeat_tensors = _flatten_tensor_output(repeated)
            if repeat_tree != first_tree:
                return VerificationCaseResult(
                    seed=seed,
                    shape=tuple(shape),
                    passed=False,
                    error_type="nondeterministic_output_tree",
                    message=f"Execution {repeat_index} returned a different output tree",
                    output_tree=repr(repeat_tree),
                    reference_tree=repr(first_tree),
                    deterministic_repeats=deterministic_repeats,
                )
            for first, repeat in zip(first_tensors, repeat_tensors, strict=True):
                if first.shape != repeat.shape or first.dtype != repeat.dtype:
                    return VerificationCaseResult(
                        seed=seed,
                        shape=tuple(shape),
                        passed=False,
                        error_type="nondeterministic_output_contract",
                        message=f"Execution {repeat_index} changed output shape or dtype",
                        deterministic_repeats=deterministic_repeats,
                    )
                finite = torch.isfinite(first) & torch.isfinite(repeat)
                masks_match = (
                    torch.equal(torch.isnan(first), torch.isnan(repeat))
                    and torch.equal(torch.isposinf(first), torch.isposinf(repeat))
                    and torch.equal(torch.isneginf(first), torch.isneginf(repeat))
                )
                if not masks_match or not torch.equal(first[finite], repeat[finite]):
                    return VerificationCaseResult(
                        seed=seed,
                        shape=tuple(shape),
                        passed=False,
                        error_type="nondeterministic_output_values",
                        message=f"Execution {repeat_index} changed output values",
                        deterministic_repeats=deterministic_repeats,
                    )
    except Exception as exc:
        return VerificationCaseResult(
            seed=seed,
            shape=tuple(shape),
            passed=False,
            error_type="nondeterministic_output_error",
            message=str(exc),
            deterministic_repeats=deterministic_repeats,
        )
    return None


def _compare_outputs(
    *,
    seed: int,
    shape: Shape,
    candidate_output: Any,
    reference_output: Any,
    candidate_inputs: tuple[Any, ...],
    reference_inputs: tuple[Any, ...],
    require_alias_contract: bool,
    deterministic_repeats: int,
    rtol: float,
    atol: float,
) -> VerificationCaseResult:
    try:
        candidate_tree, candidate_tensors = _flatten_tensor_output(candidate_output)
        reference_tree, reference_tensors = _flatten_tensor_output(reference_output)
    except Exception as exc:
        return VerificationCaseResult(
            seed=seed,
            shape=tuple(shape),
            passed=False,
            error_type="bad_output_type",
            message=str(exc),
            deterministic_repeats=deterministic_repeats,
        )

    output_shapes = [list(t.shape) for t in candidate_tensors]
    reference_shapes = [list(t.shape) for t in reference_tensors]
    output_dtypes = [str(t.dtype).replace("torch.", "") for t in candidate_tensors]
    reference_dtypes = [str(t.dtype).replace("torch.", "") for t in reference_tensors]

    if candidate_tree != reference_tree:
        return VerificationCaseResult(
            seed=seed,
            shape=tuple(shape),
            passed=False,
            error_type="output_tree_mismatch",
            message="Candidate output container structure differs from the reference",
            output_shape=output_shapes,
            reference_shape=reference_shapes,
            output_dtype=output_dtypes,
            reference_dtype=reference_dtypes,
            output_tree=repr(candidate_tree),
            reference_tree=repr(reference_tree),
            deterministic_repeats=deterministic_repeats,
        )

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
            output_tree=repr(candidate_tree),
            reference_tree=repr(reference_tree),
            deterministic_repeats=deterministic_repeats,
        )

    if require_alias_contract:
        candidate_alias = _alias_signature(candidate_tensors, candidate_inputs)
        reference_alias = _alias_signature(reference_tensors, reference_inputs)
        if candidate_alias != reference_alias:
            return VerificationCaseResult(
                seed=seed,
                shape=tuple(shape),
                passed=False,
                error_type="alias_contract_mismatch",
                message="Candidate output/input alias pattern differs from the reference",
                output_shape=output_shapes,
                reference_shape=reference_shapes,
                output_dtype=output_dtypes,
                reference_dtype=reference_dtypes,
                output_tree=repr(candidate_tree),
                reference_tree=repr(reference_tree),
                deterministic_repeats=deterministic_repeats,
                alias_contract_passed=False,
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
                output_tree=repr(candidate_tree),
                reference_tree=repr(reference_tree),
                deterministic_repeats=deterministic_repeats,
                alias_contract_passed=True if require_alias_contract else None,
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
                output_tree=repr(candidate_tree),
                reference_tree=repr(reference_tree),
                deterministic_repeats=deterministic_repeats,
                alias_contract_passed=True if require_alias_contract else None,
            )
        candidate_nan = torch.isnan(candidate)
        reference_nan = torch.isnan(reference)
        candidate_posinf = torch.isposinf(candidate)
        reference_posinf = torch.isposinf(reference)
        candidate_neginf = torch.isneginf(candidate)
        reference_neginf = torch.isneginf(reference)
        if not (
            torch.equal(candidate_nan, reference_nan)
            and torch.equal(candidate_posinf, reference_posinf)
            and torch.equal(candidate_neginf, reference_neginf)
        ):
            return VerificationCaseResult(
                seed=seed,
                shape=tuple(shape),
                passed=False,
                error_type="special_value_mask_mismatch",
                message="Candidate NaN/+Inf/-Inf masks differ from the reference",
                output_shape=output_shapes,
                reference_shape=reference_shapes,
                output_dtype=output_dtypes,
                reference_dtype=reference_dtypes,
                output_tree=repr(candidate_tree),
                reference_tree=repr(reference_tree),
                deterministic_repeats=deterministic_repeats,
                alias_contract_passed=True if require_alias_contract else None,
            )
        finite_mask = torch.isfinite(reference)
        finite_candidate = candidate[finite_mask]
        finite_reference = reference[finite_mask]
        diff = torch.abs(finite_candidate - finite_reference)
        abs_error = float(diff.max().item()) if diff.numel() else 0.0
        denominator = torch.clamp(torch.abs(finite_reference), min=1e-12)
        rel_error = float((diff / denominator).max().item()) if diff.numel() else 0.0
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)

        if not torch.allclose(
            finite_candidate,
            finite_reference,
            rtol=rtol,
            atol=atol,
        ):
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
                output_tree=repr(candidate_tree),
                reference_tree=repr(reference_tree),
                deterministic_repeats=deterministic_repeats,
                alias_contract_passed=True if require_alias_contract else None,
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
        output_tree=repr(candidate_tree),
        reference_tree=repr(reference_tree),
        deterministic_repeats=deterministic_repeats,
        alias_contract_passed=True if require_alias_contract else None,
    )


def _alias_signature(
    output_tensors: list[torch.Tensor],
    inputs: tuple[Any, ...],
) -> dict[str, list[list[bool]]]:
    input_tensors = _flatten_input_tensors(inputs)
    output_to_input = [
        [_tensors_alias(output, input_tensor) for input_tensor in input_tensors]
        for output in output_tensors
    ]
    output_to_output = [
        [_tensors_alias(left, right) for right in output_tensors]
        for left in output_tensors
    ]
    return {
        "output_to_input": output_to_input,
        "output_to_output": output_to_output,
    }


def _flatten_input_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        tensors: list[torch.Tensor] = []
        for key in sorted(value, key=str):
            tensors.extend(_flatten_input_tensors(value[key]))
        return tensors
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_flatten_input_tensors(item))
        return tensors
    return []


def _tensors_alias(left: torch.Tensor, right: torch.Tensor) -> bool:
    try:
        return bool(torch._C._is_alias_of(left, right))
    except (AttributeError, RuntimeError):
        if left.device != right.device:
            return False
        try:
            return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        except RuntimeError:
            return False
