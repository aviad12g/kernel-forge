"""Preliminary KernelBench task discovery.

The MVP smoke path does not require KernelBench. This module intentionally keeps
discovery conservative so a missing or different KernelBench checkout does not
break OpenKernelForge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelBenchTaskStub:
    task_id: str
    path: str
    name: str


@dataclass(frozen=True)
class KernelBenchDiscoveryResult:
    tasks: list[KernelBenchTaskStub]
    message: str


def discover_kernelbench_tasks(path: str | Path | None = None) -> KernelBenchDiscoveryResult:
    if path is None:
        return KernelBenchDiscoveryResult(
            tasks=[],
            message=(
                "KernelBench path was not supplied. Pass a local checkout path "
                "to discover_kernelbench_tasks(path), or add it to a future config."
            ),
        )

    root = Path(path).expanduser()
    if not root.exists():
        return KernelBenchDiscoveryResult(
            tasks=[],
            message=f"KernelBench path does not exist: {root}",
        )
    if not root.is_dir():
        return KernelBenchDiscoveryResult(
            tasks=[],
            message=f"KernelBench path is not a directory: {root}",
        )

    candidates: list[KernelBenchTaskStub] = []
    for py_file in sorted(root.rglob("*.py")):
        if py_file.name.startswith("__"):
            continue
        rel = py_file.relative_to(root)
        lowered = "/".join(rel.parts).lower()
        if "level" not in lowered and "task" not in lowered and "bench" not in lowered:
            continue
        task_id = rel.with_suffix("").as_posix().replace("/", "__")
        candidates.append(
            KernelBenchTaskStub(task_id=task_id, path=str(py_file), name=py_file.stem)
        )

    if not candidates:
        return KernelBenchDiscoveryResult(
            tasks=[],
            message=(
                f"No obvious KernelBench task files were found under {root}. "
                "This importer is preliminary; the built-in simple tasks still work."
            ),
        )

    return KernelBenchDiscoveryResult(
        tasks=candidates,
        message=f"Discovered {len(candidates)} possible KernelBench Python task files.",
    )
