from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "run_lifecycle_ablation.py"
    spec = importlib.util.spec_from_file_location("run_lifecycle_ablation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lifecycle_task_selection_covers_families_before_filling() -> None:
    module = _load_script()
    manifest = {
        "selected_task_ids": ["a0", "a1", "b0", "c0", "b1"],
        "rows": [
            {"task_id": "a0", "family": "a"},
            {"task_id": "a1", "family": "a"},
            {"task_id": "b0", "family": "b"},
            {"task_id": "c0", "family": "c"},
            {"task_id": "b1", "family": "b"},
        ],
    }
    assert module._select_lifecycle_task_ids(manifest, max_tasks=3) == ["a0", "b0", "c0"]
    assert module._select_lifecycle_task_ids(manifest, max_tasks=4) == [
        "a0",
        "b0",
        "c0",
        "a1",
    ]
