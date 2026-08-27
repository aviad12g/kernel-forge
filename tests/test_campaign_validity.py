from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "check_campaign_validity.py"
    spec = importlib.util.spec_from_file_location("check_campaign_validity", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_combined_campaign_gate_requires_both_controls(tmp_path: Path) -> None:
    module = _load_script()
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text("study:\n  id: study\n", encoding="utf-8")
    calibration_path = tmp_path / "calibration.json"
    lifecycle_path = tmp_path / "lifecycle.json"
    calibration_path.write_text('{"status":"PASS"}', encoding="utf-8")
    lifecycle_path.write_text('{"status":"PASS"}', encoding="utf-8")
    protocol = {
        "study": {"id": "study"},
        "controls": {"lifecycle": {"processes": 3}},
        "campaign_validity": {
            "failed_control_policy": "no_candidate_performance_claims",
            "lifecycle_control": {"require_complete_host_and_device_records": True},
        },
    }
    lifecycle = {
        "status": "PASS",
        "expected_tasks": 2,
        "completed_process_rows": 6,
        "median_host_lifecycle_inflation": 1.5,
        "median_enclosing_event_inflation": 1.2,
    }
    result = module.evaluate_campaign_validity(
        protocol=protocol,
        protocol_path=protocol_path,
        calibration={"status": "PASS"},
        calibration_path=calibration_path,
        lifecycle=lifecycle,
        lifecycle_path=lifecycle_path,
    )
    assert result["status"] == "PASS"

    lifecycle["completed_process_rows"] = 5
    result = module.evaluate_campaign_validity(
        protocol=protocol,
        protocol_path=protocol_path,
        calibration={"status": "PASS"},
        calibration_path=calibration_path,
        lifecycle=lifecycle,
        lifecycle_path=lifecycle_path,
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["lifecycle_complete_host_and_device_records"] is False
