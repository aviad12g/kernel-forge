"""Reports for deterministic Triton template autotune runs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from openkernelforge.reports.failure_taxonomy import classify_candidate_record
from openkernelforge.reports.run_data import load_run_bundle
from openkernelforge.reports.skipped_variants import load_skipped_variants


def write_template_autotune_report(
    run_dir: str | Path,
    *,
    compare_run_dir: str | Path | None = None,
) -> Path:
    """Write ``template_autotune_report.md`` for a template run."""

    bundle = load_run_bundle(run_dir)
    compare_bundle = load_run_bundle(compare_run_dir) if compare_run_dir else None
    path = Path(run_dir) / "template_autotune_report.md"
    path.write_text(format_template_autotune_report(bundle, compare_bundle), encoding="utf-8")
    leaderboard = template_leaderboard_rows(bundle)
    _write_leaderboard_csv(Path(run_dir) / "template_leaderboard.csv", leaderboard)
    (Path(run_dir) / "template_leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def format_template_autotune_report(
    bundle: dict[str, Any],
    compare_bundle: dict[str, Any] | None = None,
) -> str:
    run_dir = bundle["run_dir"]
    candidates = bundle["candidate_records"]
    environment = bundle.get("environment") or {}
    by_task = _by_task(candidates)
    compare_best = _best_by_task(compare_bundle["candidate_records"]) if compare_bundle else {}
    capped = [record for record in candidates if record.get("grid_was_capped")]
    skipped_variants = load_skipped_variants(run_dir)
    skipped_reasons = Counter(str(row.get("rejection_reason") or "unknown") for row in skipped_variants)
    total_possible = _first_nonempty(candidates, "total_possible_variants")
    generated = _first_nonempty(candidates, "actually_generated_variants")

    lines = [
        "# OpenKernelForge Template Autotune Report",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Environment viability: `{environment.get('viability', 'n/a')}`",
        f"- CUDA available: {'yes' if environment.get('cuda_available') else 'no'}",
        f"- Triton available: {'yes' if environment.get('triton_available') else 'no'}",
        f"- Tiny Triton kernel passed: {'yes' if environment.get('tiny_triton_kernel_passed') else 'no'}",
        f"- Variants tested: {len(candidates)}",
        f"- Total possible variants per task: {total_possible or 'n/a'}",
        f"- Actually generated variants per task: {generated or 'n/a'}",
        f"- Skipped invalid variants: {len(skipped_variants)}",
        f"- Grid capped: {'yes' if capped else 'no'}",
        "",
        "## Per-Task Best Template",
        "",
        "| Task | Candidate | BLOCK_SIZE | num_warps | num_stages | contiguous policy | allocation | shape specialized | Speedup vs eager | Speedup vs torch.compile | >=1.0x | >=0.8x | Source |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for task_id, records in sorted(by_task.items()):
        best = _best_record(records)
        benchmark = best.get("benchmark_summary") or {}
        lines.append(
            "| {task} | {candidate} | {block} | {warps} | {stages} | {policy} | {allocation} | {shape} | {eager} | {compile} | {fast} | {promising} | `{source}` |".format(
                task=task_id,
                candidate=best.get("candidate_id", "n/a"),
                block=best.get("block_size", "n/a"),
                warps=best.get("num_warps", "n/a"),
                stages=best.get("num_stages", "n/a"),
                policy=best.get("contiguous_policy", "n/a"),
                allocation=best.get("output_allocation_policy", "n/a"),
                shape="yes" if best.get("shape_specialized") else "no",
                eager=_fmt(benchmark.get("speedup_vs_eager")),
                compile=_fmt(benchmark.get("speedup_vs_torch_compile")),
                fast="yes" if _speedup_value(best) is not None and _speedup_value(best) >= 1.0 else "no",
                promising="yes" if _speedup_value(best) is not None and _speedup_value(best) >= 0.8 else "no",
                source=best.get("candidate_path", "n/a"),
            )
        )

    lines.extend(["", "## Per-Task Leaderboards", ""])
    for task_id, records in sorted(by_task.items()):
        lines.extend(
            [
                f"### {task_id}",
                "",
                "| Rank | Candidate | BLOCK_SIZE | num_warps | num_stages | contiguous policy | allocation | shape specialized | feature_dim | n_elements | Correct | Speedup vs eager | Source |",
                "| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for rank, record in enumerate(sorted(records, key=_speedup_key, reverse=True)[:10], start=1):
            benchmark = record.get("benchmark_summary") or {}
            lines.append(
                "| {rank} | {candidate} | {block} | {warps} | {stages} | {policy} | {allocation} | {shape} | {feature} | {n_mode} | {correct} | {speedup} | `{source}` |".format(
                    rank=rank,
                    candidate=record.get("candidate_id"),
                    block=record.get("block_size"),
                    warps=record.get("num_warps"),
                    stages=record.get("num_stages"),
                    policy=record.get("contiguous_policy"),
                    allocation=record.get("output_allocation_policy"),
                    shape="yes" if record.get("shape_specialized") else "no",
                    feature=record.get("feature_dim_mode"),
                    n_mode=record.get("n_elements_mode"),
                    correct="yes" if record.get("verification_passed") else "no",
                    speedup=_fmt(benchmark.get("speedup_vs_eager")),
                    source=record.get("candidate_path"),
                )
            )
        lines.append("")

    lines.extend(["", "## Speedup Quantiles", ""])
    lines.extend(["| Task | p25 | median | p75 | max | >=0.8x | >=1.0x |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for task_id, records in sorted(by_task.items()):
        values = sorted(_speedup_values(records))
        lines.append(
            "| {task} | {p25} | {median} | {p75} | {maxv} | {promising} | {fast} |".format(
                task=task_id,
                p25=_fmt(_quantile(values, 0.25)),
                median=_fmt(median(values) if values else None),
                p75=_fmt(_quantile(values, 0.75)),
                maxv=_fmt(max(values) if values else None),
                promising=sum(1 for value in values if value >= 0.8),
                fast=sum(1 for value in values if value >= 1.0),
            )
        )

    lines.extend(["", "## Variant Distributions", ""])
    for field in (
        "block_size",
        "num_warps",
        "num_stages",
        "contiguous_policy",
        "output_allocation_policy",
        "shape_specialized",
    ):
        lines.extend([f"### {field}", "", "| Value | Count | Best speedup | Median speedup |", "| --- | ---: | ---: | ---: |"])
        for value, records_for_value in sorted(_group_by_field(candidates, field).items(), key=lambda item: str(item[0])):
            values = _speedup_values(records_for_value)
            lines.append(
                f"| {value} | {len(records_for_value)} | {_fmt(max(values) if values else None)} | "
                f"{_fmt(median(values) if values else None)} |"
            )
        lines.append("")

    lines.extend(["", "## Skipped Invalid Variants", ""])
    if skipped_reasons:
        for reason, count in sorted(skipped_reasons.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")

    if compare_best:
        lines.extend(
            [
                "## Comparison Against Provided Run",
                "",
                "| Task | Template best | Comparison best | Template beats comparison | Gap |",
                "| --- | ---: | ---: | --- | ---: |",
            ]
        )
        template_best = _best_by_task(candidates)
        for task_id in sorted(set(template_best) | set(compare_best)):
            template_value = template_best.get(task_id)
            compare_value = compare_best.get(task_id)
            gap = (
                template_value - compare_value
                if template_value is not None and compare_value is not None
                else None
            )
            lines.append(
                f"| {task_id} | {_fmt(template_value)} | {_fmt(compare_value)} | "
                f"{'yes' if gap is not None and gap > 0 else 'no'} | {_fmt(gap)} |"
            )
    lines.append("")
    return "\n".join(lines)


def _by_task(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        grouped[str(record.get("task_id"))].append(record)
    return grouped


def _best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(records, key=_speedup_key) if records else {}


def _speedup_key(record: dict[str, Any]) -> tuple[int, float]:
    speedup = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
    return (1 if record.get("verification_passed") else 0, float(speedup) if speedup is not None else -1.0)


def _best_by_task(candidates: list[dict[str, Any]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for record in candidates:
        speedup = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
        if speedup is None:
            continue
        task_id = str(record.get("task_id"))
        best[task_id] = max(best.get(task_id, float("-inf")), float(speedup))
    return best


def template_leaderboard_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Return flat template leaderboard rows for CSV/JSON export."""

    rows: list[dict[str, Any]] = []
    by_task = _by_task(bundle["candidate_records"])
    for task_id, records in sorted(by_task.items()):
        for rank, record in enumerate(sorted(records, key=_speedup_key, reverse=True), start=1):
            benchmark = record.get("benchmark_summary") or {}
            classification = classify_candidate_record(record)
            rows.append(
                {
                    "task_id": task_id,
                    "leaderboard_rank": rank,
                    "candidate_path": record.get("candidate_path"),
                    "speedup_vs_eager": benchmark.get("speedup_vs_eager"),
                    "speedup_vs_torch_compile": benchmark.get("speedup_vs_torch_compile"),
                    "median_runtime": benchmark.get("candidate_median_ms"),
                    "block_size": record.get("block_size"),
                    "num_warps": record.get("num_warps"),
                    "num_stages": record.get("num_stages"),
                    "contiguous_policy": record.get("contiguous_policy"),
                    "output_allocation_policy": record.get("output_allocation_policy"),
                    "shape_specialized": record.get("shape_specialized"),
                    "feature_dim_mode": record.get("feature_dim_mode"),
                    "n_elements_mode": record.get("n_elements_mode"),
                    "correctness_passed": record.get("verification_passed"),
                    "taxonomy_label": classification.failure_type,
                }
            )
    return rows


def _write_leaderboard_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task_id",
        "leaderboard_rank",
        "candidate_path",
        "speedup_vs_eager",
        "speedup_vs_torch_compile",
        "median_runtime",
        "block_size",
        "num_warps",
        "num_stages",
        "contiguous_policy",
        "output_allocation_policy",
        "shape_specialized",
        "feature_dim_mode",
        "n_elements_mode",
        "correctness_passed",
        "taxonomy_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _first_nonempty(records: list[dict[str, Any]], key: str) -> Any:
    for record in records:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _speedup_value(record: dict[str, Any]) -> float | None:
    speedup = (record.get("benchmark_summary") or {}).get("speedup_vs_eager")
    return float(speedup) if speedup is not None else None


def _speedup_values(records: list[dict[str, Any]]) -> list[float]:
    return [value for record in records if (value := _speedup_value(record)) is not None]


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _group_by_field(candidates: list[dict[str, Any]], field: str) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        grouped[record.get(field)].append(record)
    return grouped


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
