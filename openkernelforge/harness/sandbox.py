"""Candidate file writing and trusted local import utilities.

Despite the historical module name, this module is not an operating-system
sandbox. Callers must run static policy checks first and should execute
untrusted generated kernels in a disposable worker or container.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable


class CandidateLoadError(RuntimeError):
    """Raised when a candidate file cannot be imported or has no forward."""


@dataclass(frozen=True)
class LoadedCandidate:
    path: Path
    module_name: str
    module: ModuleType
    forward: Callable | None
    model_class: type | None = None


def write_candidate_source(
    run_dir: str | Path,
    task_id: str,
    candidate_index: int,
    source: str,
) -> Path:
    if int(candidate_index) < 0:
        raise ValueError("candidate_index must be non-negative")
    candidate_dir = resolve_task_artifact_dir(run_dir, "candidates", task_id)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / f"candidate_{candidate_index:03d}.py"
    candidate_path.write_text(source.rstrip() + "\n", encoding="utf-8")
    return candidate_path


def resolve_task_artifact_dir(
    run_dir: str | Path,
    group: str,
    task_id: str,
) -> Path:
    """Resolve a task artifact directory without permitting path traversal."""

    run_root = Path(run_dir).resolve()
    group_root = (run_root / group).resolve()
    try:
        group_root.relative_to(run_root)
    except ValueError as exc:
        raise ValueError(f"Unsafe artifact group: {group!r}") from exc
    candidate_dir = (group_root / task_id).resolve()
    try:
        candidate_dir.relative_to(group_root)
    except ValueError as exc:
        raise ValueError(f"Unsafe task id for candidate artifact path: {task_id!r}") from exc
    return candidate_dir


def load_candidate_from_path(
    path: str | Path,
    *,
    require_forward: bool = True,
) -> LoadedCandidate:
    """Import a policy-checked candidate in the current process.

    This function provides deterministic module cleanup on import failure but
    does not isolate filesystem, network, process, or GPU access.
    """
    candidate_path = Path(path)
    if not candidate_path.exists():
        raise CandidateLoadError(f"Candidate file does not exist: {candidate_path}")

    module_name = f"openkernelforge_candidate_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, candidate_path)
    if spec is None or spec.loader is None:
        raise CandidateLoadError(f"Could not create import spec for {candidate_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous_dont_write_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        tb = traceback.format_exc()
        raise CandidateLoadError(f"Failed to import {candidate_path}:\n{tb}") from exc
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode

    forward = getattr(module, "forward", None)
    model_class = getattr(module, "ModelNew", None)
    if require_forward and not callable(forward):
        sys.modules.pop(module_name, None)
        raise CandidateLoadError(f"Candidate must expose callable forward(*args): {candidate_path}")
    if not require_forward and not callable(forward) and not isinstance(model_class, type):
        sys.modules.pop(module_name, None)
        raise CandidateLoadError(
            f"Candidate must expose callable forward(*args) or class ModelNew: {candidate_path}"
        )

    return LoadedCandidate(
        path=candidate_path,
        module_name=module_name,
        module=module,
        forward=forward if callable(forward) else None,
        model_class=model_class if isinstance(model_class, type) else None,
    )


def unload_candidate(candidate: LoadedCandidate) -> None:
    """Remove a loaded candidate's import registration after evaluation."""

    sys.modules.pop(candidate.module_name, None)
