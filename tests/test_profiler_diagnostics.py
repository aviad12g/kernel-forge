from scripts.profile_kernelbench_loss_candidates import _event_time, _execution_block_reason
from scripts import validate_headline_clock


def test_profiler_event_time_prefers_current_device_api():
    class Event:
        self_device_time_total = 12.5
        device_time_total = 20.0
        self_cuda_time_total = 0.0
        cuda_time_total = 0.0

    event = Event()
    assert _event_time(event, self_time=True) == 12.5
    assert _event_time(event, self_time=False) == 20.0


def test_profiler_event_time_supports_legacy_cuda_api():
    class Event:
        self_cuda_time_total = 8.0
        cuda_time_total = 10.0

    event = Event()
    assert _event_time(event, self_time=True) == 8.0
    assert _event_time(event, self_time=False) == 10.0


def test_profiler_blocks_historical_adapter_output_by_default():
    reason = _execution_block_reason(
        {
            "candidate_exists": True,
            "current_policy_passed": True,
            "current_policy_reason": "passed",
            "historical_run": True,
            "contract_recorded": False,
        },
        can_profile=True,
        environment_reason="available",
        allow_historical_debug=False,
    )
    assert reason is not None
    assert "historical adapter output" in reason


def test_profiler_requires_corrected_contract_metadata():
    reason = _execution_block_reason(
        {
            "candidate_exists": True,
            "current_policy_passed": True,
            "current_policy_reason": "passed",
            "historical_run": False,
            "contract_recorded": False,
        },
        can_profile=True,
        environment_reason="available",
        allow_historical_debug=False,
    )
    assert reason == "run record lacks corrected candidate-contract and ast-v5 policy metadata"


def test_clock_report_does_not_count_historical_replays_as_validation(tmp_path, monkeypatch):
    table = tmp_path / "headline_clock_validation.csv"
    report = tmp_path / "headline_clock_validation.md"
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr(validate_headline_clock, "TABLE", table)
    monkeypatch.setattr(validate_headline_clock, "REPORT", report)
    monkeypatch.setattr(validate_headline_clock, "ARTIFACT_DIR", artifact_dir)

    row = {
        "task": "CrossEntropyLoss",
        "source": "historical",
        "candidate_path": "candidate.py",
        "old_speedup": "1.992x",
        "old_label": "REPEAT_STABLE_WIN",
        "validation_speedup_median": "1.856x",
        "validation_per_session_speedups": "1.857x;1.856x;1.853x",
        "validation_label": "REPEAT_STABLE_WIN",
        "clocks": "clock-recorded",
        "clock_state_before": "recorded",
        "clock_state_after": "recorded",
        "temperature_power_notes": "recorded",
        "label_changed": "no",
        "status": "historical_debug_replay",
        "notes": "invalid as performance evidence",
        "evidence_status": "historical_debug_only",
    }
    validate_headline_clock._write_outputs(
        [row],
        {"cuda_available": True, "gpu_name": "GPU", "torch": "x", "triton": "y"},
        {},
        "clock-recorded",
    )

    text = report.read_text(encoding="utf-8")
    assert "Corrected validation rows: 0" in text
    assert "Historical debug replays: 1" in text
    assert "Validated rows" not in text
