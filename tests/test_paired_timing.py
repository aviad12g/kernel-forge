from __future__ import annotations

import json
import uuid

import pytest
import torch

from openkernelforge.harness.paired_timing import (
    PairedTimingConfig,
    _json_compatible_uuid,
    benchmark_paired_blocks,
    cache_buffer_size_mb,
    hash_input_snapshot,
)
from openkernelforge.tasks.simple_tasks import get_task


def test_cache_buffer_is_derived_from_l2_and_bounded() -> None:
    assert cache_buffer_size_mb(6 * 1024 * 1024, multiplier=2, minimum_mb=32, maximum_mb=512) == 32
    assert cache_buffer_size_mb(96 * 1024 * 1024, multiplier=2, minimum_mb=32, maximum_mb=512) == 192
    assert cache_buffer_size_mb(400 * 1024 * 1024, multiplier=2, minimum_mb=32, maximum_mb=512) == 512


def test_input_snapshot_hash_is_content_sensitive() -> None:
    first = (torch.tensor([1.0, 2.0]),)
    second = (torch.tensor([1.0, 3.0]),)
    assert hash_input_snapshot(first) == hash_input_snapshot(first)
    assert hash_input_snapshot(first) != hash_input_snapshot(second)


def test_gpu_uuid_is_normalized_to_json_safe_text() -> None:
    gpu_uuid = uuid.uuid4()
    payload = {"gpu_uuid": _json_compatible_uuid(gpu_uuid)}
    assert json.loads(json.dumps(payload))["gpu_uuid"] == str(gpu_uuid)


def test_paired_timing_fails_closed_without_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CPU-only gate test")
    task = get_task("vector_add")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        benchmark_paired_blocks(
            task,
            {"eager": task.reference_fn, "candidate": task.reference_fn},
            process_id="0",
            config=PairedTimingConfig(blocks=1, warmup_launches=0),
            device="cpu",
        )
