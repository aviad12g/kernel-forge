"""Candidate-budget selection optimism from independent timing artifacts."""

from __future__ import annotations

import csv
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from openkernelforge.reports.holdout_confirmation import (
    TimingBlock,
)


@dataclass(frozen=True)
class MultiplicitySummary:
    candidate_budget: int
    eligible_tasks: int
    task_resamples: int
    apparent_win_rate: float
    confirmed_win_rate: float
    median_selection_optimism_log: float
    selection_optimism_log_ci_lower: float | None
    selection_optimism_log_ci_upper: float | None
    practical_margin: float
    confirmation_rule: str = "independent_process_median_above_practical_margin"
    notes: str = ""


def analyze_selection_multiplicity(
    records: Sequence[TimingBlock],
    *,
    budgets: Sequence[int] = (1, 2, 3, 5, 10, 20),
    resamples: int = 1000,
    seed: int = 72_091,
    practical_margin: float = 0.02,
    bootstrap_samples: int = 2000,
    false_discovery_rate: float = 0.05,
    required_confirmation_processes: int = 7,
) -> list[MultiplicitySummary]:
    """Resample candidate budgets using independent data for every candidate.

    Every candidate admitted to a sampled budget must have both screening and
    fresh-process confirmation blocks. This prevents confirmation availability
    from depending on which candidate happened to win a resample. RQ2 uses the
    independent confirmation point estimate rather than repeating thousands of
    per-resample multiple-testing procedures; strict task promotion remains an
    RQ1 secondary analysis.
    """

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0 < false_discovery_rate < 1:
        raise ValueError("false_discovery_rate must be in (0, 1)")
    if required_confirmation_processes <= 0:
        raise ValueError("required_confirmation_processes must be positive")
    screening: dict[tuple[str, str], list[TimingBlock]] = {}
    confirmation: dict[tuple[str, str], list[TimingBlock]] = {}
    for record in records:
        target = screening if record.phase == "screening" else confirmation
        target.setdefault((record.task_id, record.candidate_id), []).append(record)
    tasks: dict[str, list[str]] = {}
    for key, screen_rows in screening.items():
        task_id, candidate_id = key
        confirm_rows = confirmation.get(key, [])
        process_ids = {row.process_id for row in confirm_rows if _valid(row)}
        if (
            screen_rows
            and confirm_rows
            and len(process_ids) == required_confirmation_processes
        ):
            tasks.setdefault(task_id, []).append(candidate_id)
    for candidates in tasks.values():
        candidates.sort()

    rng = random.Random(seed)
    summaries: list[MultiplicitySummary] = []
    for budget in budgets:
        if budget <= 0:
            raise ValueError("candidate budgets must be positive")
        eligible = {
            task_id: candidates
            for task_id, candidates in tasks.items()
            if len(candidates) >= budget
        }
        if not eligible:
            summaries.append(
                MultiplicitySummary(
                    candidate_budget=budget,
                    eligible_tasks=0,
                    task_resamples=0,
                    apparent_win_rate=0.0,
                    confirmed_win_rate=0.0,
                    median_selection_optimism_log=0.0,
                    selection_optimism_log_ci_lower=None,
                    selection_optimism_log_ci_upper=None,
                    practical_margin=practical_margin,
                    notes="independent all-candidate confirmation not preserved",
                )
            )
            continue

        apparent = 0
        confirmed = 0
        optimism: list[float] = []
        task_resamples = 0
        for _ in range(resamples):
            for task_id, candidates in sorted(eligible.items()):
                sampled = rng.sample(candidates, budget)
                winner = sorted(
                    sampled,
                    key=lambda candidate_id: (
                        -_screening_median(screening[(task_id, candidate_id)]),
                        candidate_id,
                    ),
                )[0]
                screen_median = _screening_median(screening[(task_id, winner)])
                confirm_median = _confirmation_median(
                    confirmation[(task_id, winner)],
                    required_processes=required_confirmation_processes,
                )
                task_resamples += 1
                if math.exp(screen_median) > 1.0 + practical_margin:
                    apparent += 1
                if math.exp(confirm_median) > 1.0 + practical_margin:
                    confirmed += 1
                optimism.append(screen_median - confirm_median)
        bootstrap_rng = random.Random(seed + budget * 1_009)
        task_ids = sorted(eligible)
        bootstrap_medians: list[float] = []
        for _ in range(bootstrap_samples):
            sampled_tasks = bootstrap_rng.choices(task_ids, k=len(task_ids))
            sampled_optimism: list[float] = []
            for task_id in sampled_tasks:
                sampled = bootstrap_rng.sample(eligible[task_id], budget)
                winner = sorted(
                    sampled,
                    key=lambda candidate_id: (
                        -_screening_median(screening[(task_id, candidate_id)]),
                        candidate_id,
                    ),
                )[0]
                sampled_optimism.append(
                    _screening_median(screening[(task_id, winner)])
                    - _confirmation_median(
                        confirmation[(task_id, winner)],
                        required_processes=required_confirmation_processes,
                    )
                )
            bootstrap_medians.append(statistics.median(sampled_optimism))
        summaries.append(
            MultiplicitySummary(
                candidate_budget=budget,
                eligible_tasks=len(eligible),
                task_resamples=task_resamples,
                apparent_win_rate=apparent / task_resamples,
                confirmed_win_rate=confirmed / task_resamples,
                median_selection_optimism_log=statistics.median(optimism) if optimism else 0.0,
                selection_optimism_log_ci_lower=_percentile(bootstrap_medians, 0.025),
                selection_optimism_log_ci_upper=_percentile(bootstrap_medians, 0.975),
                practical_margin=practical_margin,
            )
        )
    return summaries


def write_multiplicity_csv(path: str | Path, rows: Sequence[MultiplicitySummary]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(MultiplicitySummary.__annotations__)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return output


def _screening_median(records: Sequence[TimingBlock]) -> float:
    valid = [record.log_speedup for record in records if _valid(record)]
    if not valid:
        raise ValueError("screening candidate has no valid timing blocks")
    return statistics.median(valid)


def _confirmation_median(
    records: Sequence[TimingBlock],
    *,
    required_processes: int,
) -> float:
    grouped: dict[str, list[float]] = {}
    for record in records:
        if _valid(record):
            grouped.setdefault(record.process_id, []).append(record.log_speedup)
    if len(grouped) != required_processes:
        raise ValueError(
            f"confirmation process count {len(grouped)} != required {required_processes}"
        )
    process_medians = [statistics.median(grouped[key]) for key in sorted(grouped)]
    return statistics.median(process_medians)


def _valid(record: TimingBlock) -> bool:
    return record.correctness_passed and record.contract_passed


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
