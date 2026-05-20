"""Candidate file writing and local import utilities."""

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
    forward: Callable


def write_candidate_source(
    run_dir: str | Path,
    task_id: str,
    candidate_index: int,
    source: str,
) -> Path:
    candidate_dir = Path(run_dir) / "candidates" / task_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / f"candidate_{candidate_index:03d}.py"
    candidate_path.write_text(source.rstrip() + "\n", encoding="utf-8")
    return candidate_path


def load_candidate_from_path(path: str | Path) -> LoadedCandidate:
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
    if not callable(forward):
        sys.modules.pop(module_name, None)
        raise CandidateLoadError(f"Candidate must expose callable forward(*args): {candidate_path}")

    return LoadedCandidate(path=candidate_path, module_name=module_name, module=module, forward=forward)
