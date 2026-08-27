from __future__ import annotations

from openkernelforge.reports.holdout_confirmation import TimingBlock
from openkernelforge.reports.selection_multiplicity import analyze_selection_multiplicity


def _block(phase: str, candidate: str, process: int, speedup: float) -> TimingBlock:
    return TimingBlock(
        phase=phase,
        task_id="task",
        candidate_id=candidate,
        process_id=f"p{process}",
        block_id="b0",
        eager_ms=1.0,
        candidate_ms=1.0 / speedup,
    )


def test_multiplicity_uses_independent_confirmation_for_sampled_winner() -> None:
    records = [
        _block("screening", "a", 0, 1.01),
        _block("screening", "b", 0, 1.20),
    ]
    records.extend(_block("confirmation", "a", process, 1.08) for process in range(3))
    records.extend(_block("confirmation", "b", process, 0.99) for process in range(3))
    rows = analyze_selection_multiplicity(
        records,
        budgets=(2, 3),
        resamples=4,
        seed=3,
        bootstrap_samples=100,
        required_confirmation_processes=3,
    )
    assert rows[0].candidate_budget == 2
    assert rows[0].apparent_win_rate == 1.0
    assert rows[0].confirmed_win_rate == 0.0
    assert rows[0].median_selection_optimism_log > 0
    assert rows[0].selection_optimism_log_ci_lower is not None
    assert rows[0].selection_optimism_log_ci_upper is not None
    assert rows[0].selection_optimism_log_ci_lower <= rows[0].median_selection_optimism_log
    assert rows[0].selection_optimism_log_ci_upper >= rows[0].median_selection_optimism_log
    assert rows[1].notes == "independent all-candidate confirmation not preserved"
