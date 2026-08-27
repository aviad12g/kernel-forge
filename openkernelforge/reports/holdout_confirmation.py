"""Holdout-confirmation statistics for generated-kernel promotion.

This module is intentionally independent of CUDA.  It analyzes screening and
fresh-process confirmation records after correctness and contract validation.
The cluster is the OS process, not an individual CUDA-event sample.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROMOTION_LABELS = (
    "CONFIRMED_WIN",
    "SCREEN_ONLY_WIN",
    "INCONCLUSIVE",
    "BELOW_EAGER",
    "INVALID",
)


@dataclass(frozen=True)
class TimingBlock:
    phase: str
    task_id: str
    candidate_id: str
    process_id: str
    block_id: str
    eager_ms: float
    candidate_ms: float
    correctness_passed: bool = True
    contract_passed: bool = True

    @property
    def log_speedup(self) -> float:
        if self.eager_ms <= 0 or self.candidate_ms <= 0:
            raise ValueError("timings must be positive")
        return math.log(self.eager_ms / self.candidate_ms)


@dataclass(frozen=True)
class FrozenWinner:
    task_id: str
    candidate_id: str
    screening_median_log_speedup: float
    screening_speedup: float
    screening_blocks: int


@dataclass
class PromotionResult:
    task_id: str
    candidate_id: str
    screening_speedup: float | None
    confirmation_speedup: float | None
    lower_speedup_bound: float | None
    upper_speedup_bound: float | None
    practical_margin: float
    process_count: int
    block_count: int
    bootstrap_p_value: float | None
    bh_adjusted_p_value: float | None
    bh_rejected: bool
    label: str
    selection_optimism_log: float | None
    notes: str = ""


@dataclass(frozen=True)
class AggregatePromotionSummary:
    task_count: int
    valid_confirmation_tasks: int
    invalid_tasks: int
    screening_wins_above_margin: int
    confirmed_wins_above_margin: int
    screening_wins_not_confirmed: int
    false_promotion_fraction: float | None
    median_selection_optimism_log: float | None
    median_selection_optimism_ratio: float | None
    optimism_ci_lower_log: float | None
    optimism_ci_upper_log: float | None
    optimism_ci_lower_ratio: float | None
    optimism_ci_upper_ratio: float | None
    movement_p25_log: float | None
    movement_median_log: float | None
    movement_p75_log: float | None
    practical_margin: float
    bootstrap_samples: int
    bootstrap_seed: int
    interval_method: str = "task_cluster_percentile"
    notes: str = "BH-adjusted task labels are a secondary strict analysis"


def read_timing_blocks(path: str | Path) -> list[TimingBlock]:
    records: list[TimingBlock] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                TimingBlock(
                    phase=str(row["phase"]),
                    task_id=str(row["task_id"]),
                    candidate_id=str(row["candidate_id"]),
                    process_id=str(row["process_id"]),
                    block_id=str(row["block_id"]),
                    eager_ms=float(row["eager_ms"]),
                    candidate_ms=float(row["candidate_ms"]),
                    correctness_passed=_parse_bool(row.get("correctness_passed", "true")),
                    contract_passed=_parse_bool(row.get("contract_passed", "true")),
                )
            )
    return records


def select_screening_winners(records: Iterable[TimingBlock]) -> list[FrozenWinner]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in records:
        if record.phase != "screening":
            continue
        if not record.correctness_passed or not record.contract_passed:
            continue
        grouped.setdefault((record.task_id, record.candidate_id), []).append(record.log_speedup)

    by_task: dict[str, list[FrozenWinner]] = {}
    for (task_id, candidate_id), values in grouped.items():
        median_log = statistics.median(values)
        by_task.setdefault(task_id, []).append(
            FrozenWinner(
                task_id=task_id,
                candidate_id=candidate_id,
                screening_median_log_speedup=median_log,
                screening_speedup=math.exp(median_log),
                screening_blocks=len(values),
            )
        )

    winners: list[FrozenWinner] = []
    for task_id, candidates in sorted(by_task.items()):
        # Candidate ID is the deterministic tie-breaker and is recorded in the manifest.
        winners.append(
            sorted(
                candidates,
                key=lambda item: (-item.screening_median_log_speedup, item.candidate_id),
            )[0]
        )
    return winners


def analyze_holdout_confirmation(
    records: Sequence[TimingBlock],
    winners: Sequence[FrozenWinner],
    *,
    practical_margin: float = 0.02,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 62_081,
    false_discovery_rate: float = 0.05,
    required_processes: int | None = None,
) -> list[PromotionResult]:
    if practical_margin < 0:
        raise ValueError("practical_margin must be non-negative")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0 < false_discovery_rate < 1:
        raise ValueError("false_discovery_rate must be in (0, 1)")
    if required_processes is not None and required_processes <= 0:
        raise ValueError("required_processes must be positive when specified")

    margin_log = math.log1p(practical_margin)
    results: list[PromotionResult] = []
    p_values: list[tuple[int, float]] = []
    confirmation = [record for record in records if record.phase == "confirmation"]

    for winner_index, winner in enumerate(winners):
        selected = [
            record
            for record in confirmation
            if record.task_id == winner.task_id and record.candidate_id == winner.candidate_id
        ]
        valid = [
            record
            for record in selected
            if record.correctness_passed and record.contract_passed
        ]
        invalid_count = len(selected) - len(valid)
        process_values = _process_median_log_speedups(valid)
        wrong_process_count = (
            required_processes is not None and len(process_values) != required_processes
        )
        if invalid_count or not process_values or wrong_process_count:
            if invalid_count:
                note = f"{invalid_count} confirmation blocks failed correctness/contract"
            elif wrong_process_count:
                note = (
                    f"confirmation process count {len(process_values)} != "
                    f"required {required_processes}; no replacement is permitted"
                )
            else:
                note = "no confirmation process clusters"
            results.append(
                PromotionResult(
                    task_id=winner.task_id,
                    candidate_id=winner.candidate_id,
                    screening_speedup=winner.screening_speedup,
                    confirmation_speedup=None,
                    lower_speedup_bound=None,
                    upper_speedup_bound=None,
                    practical_margin=practical_margin,
                    process_count=len(process_values),
                    block_count=len(valid),
                    bootstrap_p_value=None,
                    bh_adjusted_p_value=None,
                    bh_rejected=False,
                    label="INVALID",
                    selection_optimism_log=None,
                    notes=note,
                )
            )
            continue

        seed = bootstrap_seed + winner_index * 10_007
        estimate = statistics.median(process_values)
        bootstrap_distribution = _cluster_bootstrap_medians(
            process_values,
            samples=bootstrap_samples,
            seed=seed,
        )
        alpha = 1.0 - confidence_level
        lower = _quantile(bootstrap_distribution, alpha)
        upper = _quantile(bootstrap_distribution, 1.0 - alpha)
        p_value = _centered_bootstrap_p_value(
            process_values,
            observed=estimate,
            null_value=margin_log,
            samples=bootstrap_samples,
            seed=seed + 1,
        )
        index = len(results)
        p_values.append((index, p_value))
        results.append(
            PromotionResult(
                task_id=winner.task_id,
                candidate_id=winner.candidate_id,
                screening_speedup=winner.screening_speedup,
                confirmation_speedup=math.exp(estimate),
                lower_speedup_bound=math.exp(lower),
                upper_speedup_bound=math.exp(upper),
                practical_margin=practical_margin,
                process_count=len(process_values),
                block_count=len(valid),
                bootstrap_p_value=p_value,
                bh_adjusted_p_value=None,
                bh_rejected=False,
                label="INCONCLUSIVE",
                selection_optimism_log=winner.screening_median_log_speedup - estimate,
            )
        )

    adjusted = benjamini_hochberg([value for _, value in p_values])
    for (result_index, _), adjusted_p in zip(p_values, adjusted):
        result = results[result_index]
        result.bh_adjusted_p_value = adjusted_p
        result.bh_rejected = adjusted_p <= false_discovery_rate
        lower_log = math.log(result.lower_speedup_bound or 0.0)
        upper_log = math.log(result.upper_speedup_bound or 0.0)
        if lower_log > margin_log and result.bh_rejected:
            result.label = "CONFIRMED_WIN"
        elif upper_log < 0.0:
            result.label = "BELOW_EAGER"
        elif (result.screening_speedup or 0.0) > 1.0 and upper_log <= margin_log:
            result.label = "SCREEN_ONLY_WIN"
        else:
            result.label = "INCONCLUSIVE"
    return results


def summarize_campaign_aggregates(
    results: Sequence[PromotionResult],
    *,
    practical_margin: float = 0.02,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 82_081,
) -> AggregatePromotionSummary:
    """Summarize promotion at the task level without treating blocks as IID.

    The false-promotion denominator is the prespecified set of screening
    winners whose screening speedup strictly exceeds the practical margin.
    Optimism intervals resample tasks, not CUDA-event blocks or processes.
    """

    if practical_margin < 0:
        raise ValueError("practical_margin must be non-negative")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    screen_threshold = 1.0 + practical_margin
    screening_wins = [
        result
        for result in results
        if result.screening_speedup is not None
        and result.screening_speedup > screen_threshold
    ]
    confirmed = [result for result in results if result.label == "CONFIRMED_WIN"]
    failed_confirmation = [
        result for result in screening_wins if result.label != "CONFIRMED_WIN"
    ]
    optimism = [
        float(result.selection_optimism_log)
        for result in results
        if result.selection_optimism_log is not None
    ]
    movement = [-value for value in optimism]
    optimism_median = statistics.median(optimism) if optimism else None
    interval = _task_bootstrap_median_interval(
        optimism,
        confidence_level=confidence_level,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    fraction = len(failed_confirmation) / len(screening_wins) if screening_wins else None
    invalid_count = sum(result.label == "INVALID" for result in results)
    return AggregatePromotionSummary(
        task_count=len(results),
        valid_confirmation_tasks=len(results) - invalid_count,
        invalid_tasks=invalid_count,
        screening_wins_above_margin=len(screening_wins),
        confirmed_wins_above_margin=len(confirmed),
        screening_wins_not_confirmed=len(failed_confirmation),
        false_promotion_fraction=fraction,
        median_selection_optimism_log=optimism_median,
        median_selection_optimism_ratio=(
            math.exp(optimism_median) if optimism_median is not None else None
        ),
        optimism_ci_lower_log=interval[0] if interval else None,
        optimism_ci_upper_log=interval[1] if interval else None,
        optimism_ci_lower_ratio=math.exp(interval[0]) if interval else None,
        optimism_ci_upper_ratio=math.exp(interval[1]) if interval else None,
        movement_p25_log=_quantile(movement, 0.25) if movement else None,
        movement_median_log=statistics.median(movement) if movement else None,
        movement_p75_log=_quantile(movement, 0.75) if movement else None,
        practical_margin=practical_margin,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return monotone Benjamini-Hochberg adjusted p-values."""

    count = len(p_values)
    if count == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index, value = indexed[rank_index]
        rank = rank_index + 1
        running = min(running, float(value) * count / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def write_promotion_artifacts(
    output_dir: str | Path,
    winners: Sequence[FrozenWinner],
    results: Sequence[PromotionResult],
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    winners_path = root / "screening_winners.json"
    results_json = root / "holdout_confirmation.json"
    results_csv = root / "holdout_confirmation.csv"
    report_path = root / "holdout_confirmation.md"
    winners_path.write_text(
        json.dumps([asdict(item) for item in winners], indent=2) + "\n",
        encoding="utf-8",
    )
    results_json.write_text(
        json.dumps([asdict(item) for item in results], indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(asdict(results[0]).keys()) if results else list(PromotionResult.__annotations__)
    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    lines = [
        "# Holdout Confirmation",
        "",
        "Candidate selection uses screening data; labels use fresh process-cluster confirmation data.",
        "No row in this report is valid unless correctness and contract checks passed.",
        "",
        "| Task | Candidate | Screen | Confirm | One-sided lower bound | BH p | Label |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            "| {task} | {candidate} | {screen} | {confirm} | {lower} | {p} | {label} |".format(
                task=result.task_id,
                candidate=result.candidate_id,
                screen=_format_speedup(result.screening_speedup),
                confirm=_format_speedup(result.confirmation_speedup),
                lower=_format_speedup(result.lower_speedup_bound),
                p=_format_number(result.bh_adjusted_p_value),
                label=result.label,
            )
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "winners": winners_path,
        "json": results_json,
        "csv": results_csv,
        "report": report_path,
    }


def write_aggregate_artifacts(
    output_dir: str | Path,
    summary: AggregatePromotionSummary,
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "aggregate_promotion_summary.json"
    csv_path = root / "aggregate_promotion_summary.csv"
    report_path = root / "aggregate_promotion_summary.md"
    payload = asdict(summary)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload))
        writer.writeheader()
        writer.writerow(payload)
    report_path.write_text(
        "\n".join(
            [
                "# Aggregate Holdout Outcomes",
                "",
                "Aggregate outcomes are primary; BH-adjusted task labels are secondary.",
                "",
                f"- tasks: {summary.task_count}",
                f"- valid confirmation tasks: {summary.valid_confirmation_tasks}",
                f"- screening wins above 1 + delta: {summary.screening_wins_above_margin}",
                f"- independently confirmed wins: {summary.confirmed_wins_above_margin}",
                f"- screening wins not confirmed: {summary.screening_wins_not_confirmed}",
                "- false-promotion fraction: "
                + _format_number(summary.false_promotion_fraction),
                "- median selection-optimism ratio: "
                + _format_number(summary.median_selection_optimism_ratio),
                "- task-bootstrap interval for optimism ratio: "
                + (
                    "not available"
                    if summary.optimism_ci_lower_ratio is None
                    else (
                        f"[{summary.optimism_ci_lower_ratio:.6f}, "
                        f"{summary.optimism_ci_upper_ratio:.6f}]"
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"json": json_path, "csv": csv_path, "report": report_path}


def _process_median_log_speedups(records: Sequence[TimingBlock]) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(record.process_id, []).append(record.log_speedup)
    return [statistics.median(grouped[key]) for key in sorted(grouped)]


def _cluster_bootstrap_medians(values: Sequence[float], *, samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    count = len(values)
    return [statistics.median(rng.choices(values, k=count)) for _ in range(samples)]


def _task_bootstrap_median_interval(
    values: Sequence[float],
    *,
    confidence_level: float,
    samples: int,
    seed: int,
) -> tuple[float, float] | None:
    if not values:
        return None
    distribution = _cluster_bootstrap_medians(values, samples=samples, seed=seed)
    alpha = 1.0 - confidence_level
    return (
        _quantile(distribution, alpha / 2.0),
        _quantile(distribution, 1.0 - alpha / 2.0),
    )


def _centered_bootstrap_p_value(
    values: Sequence[float],
    *,
    observed: float,
    null_value: float,
    samples: int,
    seed: int,
) -> float:
    centered = [value - observed + null_value for value in values]
    null_distribution = _cluster_bootstrap_medians(centered, samples=samples, seed=seed)
    exceedances = sum(value >= observed for value in null_distribution)
    return (exceedances + 1.0) / (samples + 1.0)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = min(max(probability, 0.0), 1.0) * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(ordered[lower_index])
    fraction = position - lower_index
    return float(ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction)


def _parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _format_speedup(value: float | None) -> str:
    return "not available" if value is None else f"{value:.4f}x"


def _format_number(value: float | None) -> str:
    return "not available" if value is None else f"{value:.6g}"
