"""Agent abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from openkernelforge.tasks.base import KernelTask


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    source: str
    metadata: dict[str, object] = field(default_factory=dict)


class KernelAgent(Protocol):
    def generate(self, task: KernelTask, *, device: str = "auto") -> CandidateSpec:
        """Return candidate source code exposing forward(*args)."""
