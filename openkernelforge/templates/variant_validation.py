"""Static validation for deterministic template variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TemplateVariantValidationResult:
    """Validation result for a deterministic template variant."""

    valid: bool
    rejection_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "rejection_reason": self.rejection_reason,
            "warnings": list(self.warnings),
        }


def validate_template_variant(metadata: dict[str, Any]) -> TemplateVariantValidationResult:
    """Validate a template variant before it enters verifier/benchmark runs."""

    block_size = metadata.get("block_size")
    try:
        block_size_int = int(block_size)
    except (TypeError, ValueError):
        return TemplateVariantValidationResult(
            valid=False,
            rejection_reason="invalid_block_size",
            warnings=[f"BLOCK_SIZE must be an integer, got {block_size!r}"],
        )

    # The current elementwise templates use tl.arange(0, BLOCK_SIZE). Triton
    # requires that arange range to be a power of two.
    if not is_power_of_two(block_size_int):
        return TemplateVariantValidationResult(
            valid=False,
            rejection_reason="block_size_not_power_of_two_for_tl_arange",
            warnings=[
                "BLOCK_SIZE must be a power of two for templates using tl.arange(0, BLOCK_SIZE)",
            ],
        )

    warnings: list[str] = []
    if block_size_int < 16:
        warnings.append("Very small BLOCK_SIZE may be dominated by launch overhead")
    return TemplateVariantValidationResult(valid=True, warnings=warnings)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0
