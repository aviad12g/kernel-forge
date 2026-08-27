from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import yaml

from openkernelforge.reports.holdout_confirmation import TimingBlock


ROOT = Path(__file__).parents[1]


def _load_script(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_delay_injection_preserves_forward_and_adds_fixed_triton_work() -> None:
    module = _load_script("freeze_near_threshold", "freeze_near_threshold_candidates.py")
    base = """import torch
import triton
import triton.language as tl


@triton.jit
def _base_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask), mask=mask)


def forward(*args):
    x, = args
    output = torch.empty_like(x)
    _base_kernel[(1,)](x, output, x.numel(), BLOCK_SIZE=256)
    return output
"""

    source = module.inject_discarded_copy_work(
        base,
        work_units=1.25,
        block_size=256,
        num_warps=4,
    )

    ast.parse(source)
    assert source.count("_okf_discarded_copy_kernel[(") == 2
    assert "_okf_delay_tail = (_okf_delay_n * 250 + 999) // 1000" in source
    assert "return output" in source


def test_near_threshold_selection_prefers_window_then_closest() -> None:
    module = _load_script(
        "run_near_threshold",
        "run_near_threshold_multiplicity_campaign.py",
    )
    selected, reasons = module.select_calibrated_candidates(
        {
            "inside_low": 0.99,
            "inside_high": 1.03,
            "closest_outside": 1.05,
            "far": 1.40,
        },
        count=3,
        lower=0.98,
        upper=1.04,
    )

    assert set(selected[:2]) == {"inside_low", "inside_high"}
    assert selected[2] == "closest_outside"
    assert reasons["closest_outside"] == "closest_absolute_log_distance_fallback"


def test_near_threshold_protocol_freezes_twenty_then_selects_twelve() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs" / "workshop2026_near_threshold_multiplicity_protocol.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert protocol["candidates"]["variants_per_task"] == 20
    assert protocol["candidates"]["selected_candidates_per_task"] == 12
    assert protocol["calibration"]["excluded_from_primary_analysis"] is True
    assert protocol["confirmation"]["fresh_processes"] == 7
    assert protocol["budget"]["maximum_recorded_gpu_worker_hours"] == 5.0
    for task_id in protocol["tasks"]["ids"]:
        assert len(protocol["candidates"]["delay_work_units"][task_id]) == 20


def test_v2_protocol_requires_twelve_candidates_inside_window() -> None:
    protocol = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "workshop2026_near_threshold_multiplicity_v2_protocol.yaml"
        ).read_text(encoding="utf-8")
    )

    assert protocol["candidates"]["variants_per_task"] == 20
    assert protocol["candidates"]["selected_candidates_per_task"] == 12
    assert protocol["calibration"]["minimum_candidates_in_window_per_task_to_advance"] == 12
    assert protocol["budget"]["maximum_recorded_gpu_worker_hours"] == 4.7


def test_v3_protocol_uses_supported_eight_candidate_budget() -> None:
    protocol = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "workshop2026_near_threshold_multiplicity_v3_protocol.yaml"
        ).read_text(encoding="utf-8")
    )

    assert protocol["candidates"]["selected_candidates_per_task"] == 8
    assert protocol["calibration"]["minimum_candidates_in_window_per_task_to_advance"] == 8
    assert protocol["analysis"]["candidate_budgets"] == [1, 2, 3, 5, 8]


def test_near_threshold_postprocessor_requires_complete_process_coverage() -> None:
    module = _load_script(
        "analyze_near_threshold",
        "analyze_near_threshold_campaign.py",
    )
    records = [
        TimingBlock(
            phase="screening",
            task_id="bias_relu",
            candidate_id="delay_00",
            process_id="screening",
            block_id=str(index),
            eager_ms=1.0,
            candidate_ms=1.0,
        )
        for index in range(2)
    ]
    records.extend(
        TimingBlock(
            phase="confirmation",
            task_id="bias_relu",
            candidate_id="delay_00",
            process_id=f"p{process_index:02d}",
            block_id=str(block_index),
            eager_ms=1.0,
            candidate_ms=1.0,
        )
        for process_index in range(2)
        for block_index in range(2)
    )

    module._verify_coverage(
        records,
        {"bias_relu": ["delay_00"]},
        processes=2,
        blocks_per_process=2,
    )

    records.pop()
    try:
        module._verify_coverage(
            records,
            {"bias_relu": ["delay_00"]},
            processes=2,
            blocks_per_process=2,
        )
    except RuntimeError as error:
        assert "confirmation coverage is incomplete" in str(error)
    else:
        raise AssertionError("incomplete confirmation coverage was accepted")
