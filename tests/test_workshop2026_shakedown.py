from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "run_workshop2026_shakedown.py"
    spec = importlib.util.spec_from_file_location("run_workshop2026_shakedown", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shakedown_selects_only_feasible_excluded_task() -> None:
    module = _load_script()
    manifest = {
        "selected_task_ids": ["selected"],
        "rows": [
            {"task_id": "selected", "feasible": True, "source_relative_path": "001.py"},
            {"task_id": "bad", "feasible": False, "source_relative_path": "002.py"},
            {"task_id": "later", "feasible": True, "source_relative_path": "004.py"},
            {"task_id": "first", "feasible": True, "source_relative_path": "003.py"},
        ],
    }
    assert module.select_excluded_shakedown_task(manifest) == "first"


def test_shakedown_fails_closed_without_linux_cuda() -> None:
    module = _load_script()
    with pytest.raises(RuntimeError, match="Linux CUDA/Triton"):
        module._require_cuda_linux()
