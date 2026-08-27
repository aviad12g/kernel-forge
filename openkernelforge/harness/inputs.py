"""Input-tree helpers shared by verification and benchmarking."""

from __future__ import annotations

from typing import Any

import torch


def clone_value(value: Any) -> Any:
    """Clone tensors recursively while preserving container structure."""

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [clone_value(child) for child in value]
    if isinstance(value, tuple):
        if hasattr(value, "_fields"):
            return type(value)(*(clone_value(child) for child in value))
        return tuple(clone_value(child) for child in value)
    return value


def clone_inputs(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(clone_value(item) for item in inputs)


def find_value_difference(before: Any, after: Any, *, path: str = "inputs") -> str | None:
    """Return the first differing path in two nested input trees."""

    if isinstance(before, torch.Tensor) or isinstance(after, torch.Tensor):
        if not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor):
            return path
        if (
            before.shape != after.shape
            or before.dtype != after.dtype
            or before.device != after.device
        ):
            return path
        if torch.equal(before, after):
            return None
        if (before.is_floating_point() or before.is_complex()) and torch.allclose(
            before,
            after,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            return None
        return path

    if isinstance(before, dict) or isinstance(after, dict):
        if not isinstance(before, dict) or not isinstance(after, dict):
            return path
        if set(before) != set(after):
            return path
        for key in before:
            difference = find_value_difference(
                before[key],
                after[key],
                path=f"{path}.{key}",
            )
            if difference:
                return difference
        return None

    if isinstance(before, (list, tuple)) or isinstance(after, (list, tuple)):
        if type(before) is not type(after) or len(before) != len(after):
            return path
        for index, (before_item, after_item) in enumerate(zip(before, after, strict=True)):
            difference = find_value_difference(
                before_item,
                after_item,
                path=f"{path}[{index}]",
            )
            if difference:
                return difference
        return None

    try:
        return None if before == after else path
    except Exception:
        return path if before is not after else None
