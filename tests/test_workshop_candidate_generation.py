from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from openkernelforge.agents.backends import FakeBackend
from openkernelforge.tasks.base import KernelTask, TaskTolerance


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "generate_workshop2026_candidates.py"
    spec = importlib.util.spec_from_file_location("generate_workshop2026_candidates", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task(tmp_path: Path) -> KernelTask:
    source_path = tmp_path / "task.py"
    source_path.write_text("def reference_fn(x):\n    return x\n", encoding="utf-8")
    return KernelTask(
        task_id="vector_add",
        name="vector add",
        description="test",
        reference_fn=lambda x, y: x + y,
        input_generator=lambda seed, shape, dtype, device: (
            torch.ones(shape, dtype=dtype, device=device),
            torch.ones(shape, dtype=dtype, device=device),
        ),
        allowed_dtypes=(torch.float32,),
        tolerance=TaskTolerance(),
        benchmark_shapes=[(4,)],
        reference_source="def forward(x, y): return x + y",
        metadata={
            "source_path": str(source_path),
            "candidate_contract": "forward",
            "reference_has_model_state": False,
            "op_family": "activation",
            "shape_metadata": {"shape": [4]},
        },
    )


def test_generation_preserves_all_manifest_artifacts(tmp_path: Path) -> None:
    module = _load_script()
    generation = {
        "prompt_version": "workshop2026_v1",
        "configured_model_string": "fake-deterministic",
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 128,
        "provider": "fake",
        "candidates_per_task": 3,
        "max_attempts_per_candidate": 1,
    }
    record = module._generate_one(
        FakeBackend(mode="correct"),
        _task(tmp_path),
        candidate_index=0,
        output_root=tmp_path / "candidates",
        generation=generation,
    )
    for field in ("path", "prompt_path", "raw_response_path", "metadata_path"):
        assert Path(record[field]).exists()
    assert record["provider_response_model"] == "fake-deterministic"


def test_manifest_marks_missing_provider_model_metadata() -> None:
    module = _load_script()
    generation = {
        "provider": "test",
        "configured_model_string": "configured",
        "prompt_version": "v1",
        "candidates_per_task": 3,
        "max_attempts_per_candidate": 1,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 128,
    }
    payload = module._manifest_payload(
        task_manifest_sha="sha",
        generation=generation,
        records={"task": [{"provider_response_model": "not_returned"}]},
    )
    assert payload["provider_response_model_fields_preserved"] is False
