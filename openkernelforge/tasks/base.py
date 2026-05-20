"""Core task abstraction for kernel-generation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

Shape = tuple[int, ...]
ReferenceFn = Callable[..., Any]
InputGenerator = Callable[[int, Shape, torch.dtype, torch.device], tuple[Any, ...]]


@dataclass(frozen=True)
class TaskTolerance:
    rtol: float = 1e-4
    atol: float = 1e-5


@dataclass(frozen=True)
class KernelTask:
    task_id: str
    name: str
    description: str
    reference_fn: ReferenceFn
    input_generator: InputGenerator
    allowed_dtypes: tuple[torch.dtype, ...]
    tolerance: TaskTolerance
    benchmark_shapes: list[Shape]
    reference_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def generate_inputs(
        self,
        seed: int,
        shape: Shape | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[Any, ...]:
        selected_shape = shape or self.benchmark_shapes[0]
        selected_dtype = dtype or self.allowed_dtypes[0]
        selected_device = torch.device(device or "cpu")

        if selected_dtype not in self.allowed_dtypes:
            allowed = ", ".join(str(d).replace("torch.", "") for d in self.allowed_dtypes)
            raise ValueError(
                f"Task {self.task_id} does not allow dtype "
                f"{str(selected_dtype).replace('torch.', '')}; allowed: {allowed}"
            )

        return self.input_generator(seed, selected_shape, selected_dtype, selected_device)

    def prompt_metadata(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "allowed_dtypes": [str(dtype).replace("torch.", "") for dtype in self.allowed_dtypes],
            "rtol": self.tolerance.rtol,
            "atol": self.tolerance.atol,
            "benchmark_shapes": [list(shape) for shape in self.benchmark_shapes],
            "reference_source": self.reference_source,
            "metadata": self.metadata,
        }
