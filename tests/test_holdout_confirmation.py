from __future__ import annotations

import math

from openkernelforge.reports.holdout_confirmation import (
    TimingBlock,
    analyze_holdout_confirmation,
    benjamini_hochberg,
    select_screening_winners,
    summarize_campaign_aggregates,
)


def _block(
    phase: str,
    task: str,
    candidate: str,
    process: int,
    block: int,
    speedup: float,
    *,
    valid: bool = True,
) -> TimingBlock:
    return TimingBlock(
        phase=phase,
        task_id=task,
        candidate_id=candidate,
        process_id=str(process),
        block_id=str(block),
        eager_ms=1.0,
        candidate_ms=1.0 / speedup,
        correctness_passed=valid,
        contract_passed=valid,
    )


def test_screening_selects_best_candidate_with_stable_tie_break() -> None:
    records = [
        _block("screening", "task", "candidate_b", 0, 0, 1.1),
        _block("screening", "task", "candidate_a", 0, 0, 1.1),
        _block("screening", "task", "candidate_c", 0, 0, 1.2, valid=False),
    ]
    winners = select_screening_winners(records)
    assert len(winners) == 1
    assert winners[0].candidate_id == "candidate_a"
    assert math.isclose(winners[0].screening_speedup, 1.1)


def test_clear_fresh_process_win_is_confirmed() -> None:
    records = [_block("screening", "task", "candidate", 0, block, 1.12) for block in range(10)]
    records += [
        _block("confirmation", "task", "candidate", process, block, 1.08 + process * 0.001)
        for process in range(7)
        for block in range(5)
    ]
    winners = select_screening_winners(records)
    result = analyze_holdout_confirmation(
        records,
        winners,
        bootstrap_samples=2000,
        bootstrap_seed=7,
    )[0]
    assert result.process_count == 7
    assert result.label == "CONFIRMED_WIN"
    assert result.lower_speedup_bound is not None and result.lower_speedup_bound > 1.02


def test_screening_win_that_confirmation_excludes_is_screen_only() -> None:
    records = [_block("screening", "task", "candidate", 0, block, 1.08) for block in range(10)]
    records += [
        _block("confirmation", "task", "candidate", process, block, 1.005)
        for process in range(7)
        for block in range(5)
    ]
    winners = select_screening_winners(records)
    result = analyze_holdout_confirmation(
        records,
        winners,
        bootstrap_samples=1000,
        bootstrap_seed=11,
    )[0]
    assert result.label == "SCREEN_ONLY_WIN"


def test_invalid_confirmation_cannot_be_promoted() -> None:
    records = [_block("screening", "task", "candidate", 0, 0, 1.2)]
    records.append(_block("confirmation", "task", "candidate", 0, 0, 1.2, valid=False))
    result = analyze_holdout_confirmation(
        records,
        select_screening_winners(records),
        bootstrap_samples=100,
    )[0]
    assert result.label == "INVALID"


def test_required_process_count_fails_closed_without_replacement() -> None:
    records = [_block("screening", "task", "candidate", 0, 0, 1.2)]
    records.extend(
        _block("confirmation", "task", "candidate", process, 0, 1.1)
        for process in range(6)
    )
    result = analyze_holdout_confirmation(
        records,
        select_screening_winners(records),
        bootstrap_samples=100,
        required_processes=7,
    )[0]
    assert result.label == "INVALID"
    assert "no replacement" in result.notes


def test_aggregate_summary_uses_margin_and_task_bootstrap() -> None:
    records = []
    for task, screen, confirm in [
        ("confirmed", 1.20, 1.10),
        ("false_promotion", 1.15, 1.00),
        ("not_screen_win", 1.01, 1.00),
    ]:
        records.extend(_block("screening", task, "c", 0, block, screen) for block in range(3))
        records.extend(
            _block("confirmation", task, "c", process, block, confirm)
            for process in range(7)
            for block in range(3)
        )
    results = analyze_holdout_confirmation(
        records,
        select_screening_winners(records),
        bootstrap_samples=500,
        required_processes=7,
    )
    summary = summarize_campaign_aggregates(
        results,
        bootstrap_samples=500,
        bootstrap_seed=9,
    )
    assert summary.task_count == 3
    assert summary.screening_wins_above_margin == 2
    assert summary.confirmed_wins_above_margin == 1
    assert summary.screening_wins_not_confirmed == 1
    assert summary.false_promotion_fraction == 0.5
    assert summary.optimism_ci_lower_log is not None


def test_benjamini_hochberg_is_monotone_in_sorted_order() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.002])
    assert adjusted == [0.02, 0.04, 0.04, 0.008]
