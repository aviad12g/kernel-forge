from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_evaluator_controls_fail_closed_without_linux_cuda() -> None:
    path = Path(__file__).parents[1] / "scripts" / "run_evaluator_controls.py"
    spec = importlib.util.spec_from_file_location("run_evaluator_controls", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match="Linux CUDA/Triton"):
        module._require_cuda_linux()


def test_calibration_validity_applies_preregistered_thresholds() -> None:
    path = Path(__file__).parents[1] / "scripts" / "run_evaluator_controls.py"
    spec = importlib.util.spec_from_file_location("run_evaluator_controls_validity", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.evaluate_calibration_validity(
        {
            "null_wrapper_over_persistent": {
                "processes": 7,
                "median_process_ratio": 1.001,
            },
            "positive_over_persistent": {
                "processes": 7,
                "median_process_ratio": 1.0 / 1.04,
            },
        },
        expected_processes=7,
        validity_config={
            "practical_speedup_margin": 0.02,
            "null_control": {"median_ratio_bounds": [0.995, 1.005]},
            "known_slowdown_control": {
                "minimum_detected_slowdown": 0.02,
                "maximum_detected_slowdown": 0.08,
            },
        },
    )
    assert result["status"] == "PASS"
    assert all(result["checks"].values())


def test_calibration_validity_fails_on_missing_process() -> None:
    path = Path(__file__).parents[1] / "scripts" / "run_evaluator_controls.py"
    spec = importlib.util.spec_from_file_location("run_evaluator_controls_missing", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.evaluate_calibration_validity(
        {
            "null_wrapper_over_persistent": {
                "processes": 6,
                "median_process_ratio": 1.0,
            },
            "positive_over_persistent": {
                "processes": 7,
                "median_process_ratio": 1.0 / 1.04,
            },
        },
        expected_processes=7,
        validity_config={
            "practical_speedup_margin": 0.02,
            "null_control": {"median_ratio_bounds": [0.995, 1.005]},
            "known_slowdown_control": {
                "minimum_detected_slowdown": 0.02,
                "maximum_detected_slowdown": 0.08,
            },
        },
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["exact_process_count"] is False


def test_empirical_extra_work_calibration_is_bounded() -> None:
    path = Path(__file__).parents[1] / "scripts" / "run_evaluator_controls.py"
    spec = importlib.util.spec_from_file_location("run_evaluator_controls_sleep", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows, ratio = module.select_empirical_extra_rows(
        lambda value: 1.0 + value / 100.0,
        maximum_rows=64,
        target_ratio=1.04,
    )

    assert rows == 4
    assert ratio == pytest.approx(1.04)


def test_control_workload_is_large_enough_for_small_slowdown_calibration() -> None:
    path = Path(__file__).parents[1] / "scripts" / "run_evaluator_controls.py"
    spec = importlib.util.spec_from_file_location("run_evaluator_controls_shape", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._control_task().benchmark_shapes == [
        (module.CONTROL_ROWS, module.CONTROL_FEATURES)
    ]
    assert module.CONTROL_FEATURES == 4096
