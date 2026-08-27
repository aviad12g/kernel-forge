"""Deterministic template agent for Triton baseline/autotune runs."""

from __future__ import annotations

from openkernelforge.agents.base import CandidateSpec
from openkernelforge.tasks.base import KernelTask
from openkernelforge.templates.elementwise_templates import (
    DEFAULT_BLOCK_SIZES,
    DEFAULT_CONTIGUOUS_POLICIES,
    DEFAULT_NUM_WARPS,
    DEFAULT_NUM_STAGES,
    DEFAULT_OUTPUT_ALLOCATION_POLICIES,
    DEFAULT_N_ELEMENTS_MODES,
    DEFAULT_FEATURE_DIM_MODES,
    generate_elementwise_templates,
)
from openkernelforge.templates.fused_templates import (
    generate_fused_templates,
)
from openkernelforge.templates.variant_validation import validate_template_variant


class TemplateAgent:
    """Generate deterministic Triton template candidates without an LLM."""

    def __init__(
        self,
        *,
        template_family: str = "elementwise",
        template_variants: dict[str, object] | None = None,
    ) -> None:
        self.template_family = template_family
        self.template_variants = template_variants or {}
        self.per_task_variants = _dict_value(self.template_variants.get("per_task"))
        self.generation_stage = str(
            self.template_variants.get("generation_stage", "template_baseline")
        )
        self.focused_seed_run = _optional_str(self.template_variants.get("focused_seed_run"))
        self.focused_seed_candidates = _dict_value(
            self.template_variants.get("focused_seed_candidates")
        )
        self.skipped_variants_by_task: dict[str, list[dict[str, object]]] = {}

    def generate_all(self, task: KernelTask) -> list[CandidateSpec]:
        """Return all template variants for a supported task."""

        if self.template_family not in {"elementwise", "fused8"}:
            raise ValueError(f"Unknown template family: {self.template_family}")
        variants = self._variants_for_task(task.task_id)
        block_sizes = _int_list(variants.get("block_sizes"), DEFAULT_BLOCK_SIZES)
        num_warps = _int_list(variants.get("num_warps"), DEFAULT_NUM_WARPS)
        num_stages = _int_list(variants.get("num_stages"), DEFAULT_NUM_STAGES)
        contiguous_policies = _str_list(
            variants.get("contiguous_policies"),
            DEFAULT_CONTIGUOUS_POLICIES,
        )
        output_allocation_policies = _str_list(
            variants.get("output_allocation_policies"),
            DEFAULT_OUTPUT_ALLOCATION_POLICIES,
        )
        n_elements_modes = _str_list(
            variants.get("n_elements_modes"),
            DEFAULT_N_ELEMENTS_MODES,
        )
        feature_dim_modes = _str_list(
            variants.get("feature_dim_modes"),
            DEFAULT_FEATURE_DIM_MODES,
        )
        reduction_block_sizes = (
            _int_list(variants.get("reduction_block_sizes"), tuple(block_sizes))
            if variants.get("reduction_block_sizes") is not None
            else None
        )
        max_variants_per_task = _optional_int(variants.get("max_variants_per_task"), 200)
        grid_sampling = str(variants.get("grid_sampling", "capped_ordered"))
        sort_order = str(variants.get("sort_order", "small_to_large"))
        validate_variants = bool(variants.get("validate_variants", False))
        record_skipped_variants = bool(variants.get("record_skipped_variants", False))
        if self.template_family == "elementwise":
            candidates = generate_elementwise_templates(
                task.task_id,
                block_sizes=block_sizes,
                num_warps=num_warps,
                num_stages=num_stages,
                contiguous_policies=contiguous_policies,
                output_allocation_policies=output_allocation_policies,
                n_elements_modes=n_elements_modes,
                feature_dim_modes=feature_dim_modes,
            )
        else:
            candidates = generate_fused_templates(
                task.task_id,
                block_sizes=block_sizes,
                reduction_block_sizes=reduction_block_sizes,
                num_warps=num_warps,
                num_stages=num_stages,
                contiguous_policies=contiguous_policies,
                output_allocation_policies=output_allocation_policies,
                n_elements_modes=n_elements_modes,
                feature_dim_modes=feature_dim_modes,
            )
        total_possible = len(candidates)
        valid_candidates: list[CandidateSpec] = []
        skipped: list[dict[str, object]] = []
        for candidate in candidates:
            validation = validate_template_variant(candidate.metadata) if validate_variants else None
            if validation is None or validation.valid:
                valid_candidates.append(
                    CandidateSpec(
                        name=candidate.name,
                        source=candidate.source,
                        metadata={
                            **candidate.metadata,
                            "variant_validation": validation.to_dict() if validation else None,
                        },
                    )
                )
            else:
                skipped_item: dict[str, object] = {
                    "task_id": task.task_id,
                    "candidate_name": candidate.name,
                    "template_metadata": dict(candidate.metadata),
                    "rejection_reason": validation.rejection_reason,
                    "warnings": list(validation.warnings),
                    "source_type": "template_variant_validation",
                }
                skipped.append(skipped_item)
        if record_skipped_variants:
            self.skipped_variants_by_task[task.task_id] = skipped
        ordered = _order_candidates(valid_candidates, sort_order)
        valid_possible = len(valid_candidates)
        if grid_sampling == "exhaustive" and valid_possible <= max_variants_per_task:
            selected = ordered
        elif grid_sampling == "exhaustive" and valid_possible > max_variants_per_task:
            selected = ordered[:max_variants_per_task]
        elif grid_sampling in {"capped_ordered", "exhaustive_if_under_cap", "auto"}:
            selected = ordered if valid_possible <= max_variants_per_task else ordered[:max_variants_per_task]
        else:
            raise ValueError(f"Unknown template grid_sampling: {grid_sampling}")
        skipped_reasons = _reason_counts(skipped)
        return [
            CandidateSpec(
                name=candidate.name,
                source=candidate.source,
                metadata={
                    **candidate.metadata,
                    "generation_stage": self.generation_stage,
                    "total_possible_variants": total_possible,
                    "valid_possible_variants": valid_possible,
                    "generated_valid_variants": len(selected),
                    "skipped_invalid_variants": len(skipped),
                    "skipped_reasons": skipped_reasons,
                    "actually_generated_variants": len(selected),
                    "grid_sampling": grid_sampling,
                    "sort_order": sort_order,
                    "grid_was_capped": len(selected) < valid_possible,
                    "focused_seed_run": self.focused_seed_run,
                    "focused_seed_candidate_path": self.focused_seed_candidates.get(task.task_id),
                },
            )
            for candidate in selected
        ]

    def _variants_for_task(self, task_id: str) -> dict[str, object]:
        variants = {
            key: value
            for key, value in self.template_variants.items()
            if key not in {"per_task", "focused_seed_candidates"}
        }
        task_overrides = self.per_task_variants.get(task_id)
        if isinstance(task_overrides, dict):
            variants.update(task_overrides)
        return variants


def _int_list(value: object, default: tuple[int, ...]) -> list[int]:
    if value is None:
        return list(default)
    if not isinstance(value, list):
        raise ValueError("Template variant field must be a list")
    return [int(item) for item in value]


def _str_list(value: object, default: tuple[str, ...]) -> list[str]:
    if value is None:
        return list(default)
    if not isinstance(value, list):
        raise ValueError("Template variant field must be a list")
    return [str(item) for item in value]


def _optional_int(value: object, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, (str, int)):
        raise TypeError(f"Expected an integer template option, got {type(value).__name__}")
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _dict_value(value: object) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _reason_counts(skipped: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in skipped:
        reason = str(item.get("rejection_reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _order_candidates(candidates: list[CandidateSpec], sort_order: str) -> list[CandidateSpec]:
    if sort_order == "small_to_large":
        return sorted(candidates, key=_candidate_order_key)
    if sort_order == "large_to_small":
        return sorted(candidates, key=_candidate_order_key, reverse=True)
    if sort_order == "mixed":
        ordered = sorted(candidates, key=_candidate_order_key)
        result: list[CandidateSpec] = []
        left = 0
        right = len(ordered) - 1
        while left <= right:
            result.append(ordered[left])
            if left != right:
                result.append(ordered[right])
            left += 1
            right -= 1
        return result
    raise ValueError(f"Unknown template sort_order: {sort_order}")


def _candidate_order_key(candidate: CandidateSpec) -> tuple:
    metadata = candidate.metadata
    return (
        _optional_int(metadata.get("block_size"), 0),
        _optional_int(metadata.get("num_warps"), 0),
        _optional_int(metadata.get("num_stages"), 0),
        str(metadata.get("contiguous_policy", "")),
        str(metadata.get("output_allocation_policy", "")),
        str(metadata.get("n_elements_mode", "")),
        str(metadata.get("feature_dim_mode", "")),
        _optional_int(metadata.get("reduction_block_size"), 0),
    )
