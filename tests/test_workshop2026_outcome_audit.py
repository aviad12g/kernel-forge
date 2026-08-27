from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _load_script():
    path = ROOT / "scripts" / "summarize_workshop2026_outcomes.py"
    spec = importlib.util.spec_from_file_location("summarize_workshop2026_outcomes", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corrected_candidate_funnel_and_compiler_rung_match_frozen_records() -> None:
    module = _load_script()
    campaign = ROOT / "artifacts" / "workshop2026" / "holdout_campaign"
    failure_rows = module.summarize_failures(campaign)
    counts = {row["category"]: row["candidates"] for row in failure_rows}
    assert counts == {
        "static_policy_failure": 77,
        "contract_failure": 0,
        "correctness_failure": 9,
        "candidate_compile_or_runtime_failure": 28,
        "full_gate_pass": 27,
        "not_evaluated_compiler_baseline_failure": 3,
    }

    compiler_rows = module.summarize_compiler_rung(campaign)
    screening = next(
        row
        for row in compiler_rows
        if row["scope"] == "frozen_valid_task_winners_screening"
    )
    assert screening["units"] == 10
    assert screening["compiler_available"] == 10
    assert screening["above_compile_1_02"] == 1
    assert screening["above_eager_1_02"] == 0
