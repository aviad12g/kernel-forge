#!/usr/bin/env python3
"""Validate and summarize a completed near-threshold multiplicity campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

from openkernelforge.reports.holdout_confirmation import (
    read_timing_blocks,
    select_screening_winners,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign-dir",
        default="artifacts/workshop2026/near_threshold_multiplicity_v3/campaign",
    )
    parser.add_argument(
        "--selected-manifest",
        default=(
            "artifacts/workshop2026/near_threshold_multiplicity_v3/"
            "selected_candidate_manifest.json"
        ),
    )
    parser.add_argument(
        "--multiplicity-output",
        default="reports/tables/workshop2026_near_threshold_multiplicity.csv",
    )
    parser.add_argument(
        "--winners-output",
        default="reports/tables/workshop2026_near_threshold_winners.csv",
    )
    parser.add_argument(
        "--report-output",
        default="reports/workshop2026_near_threshold_multiplicity.md",
    )
    args = parser.parse_args()

    campaign = _resolve(args.campaign_dir)
    selected_path = _resolve(args.selected_manifest)
    multiplicity_output = _resolve(args.multiplicity_output)
    winners_output = _resolve(args.winners_output)
    report_output = _resolve(args.report_output)

    _verify_ledger(campaign)
    state = json.loads((campaign / "campaign_state.json").read_text(encoding="utf-8"))
    if state.get("state") != "complete":
        raise RuntimeError(f"campaign is not complete: {state.get('state')!r}")

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected.get("status") != "FROZEN_AFTER_DISJOINT_CALIBRATION_BEFORE_PRIMARY_TIMING":
        raise RuntimeError("selected manifest does not have the frozen primary status")
    if not selected.get("calibration_excluded_from_primary_analysis"):
        raise RuntimeError("calibration is not marked as excluded from primary analysis")
    if any(
        reason != "within_preregistered_window"
        for reasons in selected["selection_reasons"].values()
        for reason in reasons.values()
    ):
        raise RuntimeError("selected manifest contains a calibration-window fallback")

    blocks_path = campaign / "primary_timing_blocks.csv"
    records = read_timing_blocks(blocks_path)
    expected_tasks = sorted(selected["tasks"])
    expected_candidates = {
        task_id: sorted(item["candidate_id"] for item in selected["tasks"][task_id])
        for task_id in expected_tasks
    }
    _verify_coverage(records, expected_candidates, processes=7, blocks_per_process=20)

    winners = select_screening_winners(records)
    winner_rows: list[dict[str, object]] = []
    margin = 0.02
    for winner in winners:
        confirmation = [
            row
            for row in records
            if row.phase == "confirmation"
            and row.task_id == winner.task_id
            and row.candidate_id == winner.candidate_id
        ]
        grouped: dict[str, list[float]] = {}
        for row in confirmation:
            grouped.setdefault(row.process_id, []).append(row.log_speedup)
        process_medians = [statistics.median(grouped[key]) for key in sorted(grouped)]
        confirmation_log = statistics.median(process_medians)
        confirmation_speedup = math.exp(confirmation_log)
        screen_win = winner.screening_speedup > 1.0 + margin
        confirm_win = confirmation_speedup > 1.0 + margin
        label = (
            "CONFIRMED_WIN"
            if screen_win and confirm_win
            else "SCREEN_ONLY_WIN"
            if screen_win
            else "CONFIRM_ONLY_WIN"
            if confirm_win
            else "BELOW_MARGIN"
        )
        winner_rows.append(
            {
                "task": winner.task_id,
                "winner_candidate": winner.candidate_id,
                "candidate_budget": len(expected_candidates[winner.task_id]),
                "screening_speedup": winner.screening_speedup,
                "confirmation_speedup": confirmation_speedup,
                "selection_optimism_log": (
                    winner.screening_median_log_speedup - confirmation_log
                ),
                "screening_above_margin": screen_win,
                "confirmation_above_margin": confirm_win,
                "label": label,
                "confirmation_processes": len(grouped),
                "confirmation_blocks": len(confirmation),
                "per_process_speedups": ";".join(
                    f"{math.exp(value):.9f}" for value in process_medians
                ),
            }
        )

    multiplicity_source = campaign / "selection_multiplicity.csv"
    multiplicity_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(multiplicity_source, multiplicity_output)
    _write_csv(winners_output, winner_rows)
    _write_report(
        report_output,
        state=state,
        selected=selected,
        winners=winner_rows,
        multiplicity_path=multiplicity_output,
        environment=_read_environment(campaign),
    )
    print(f"near-threshold multiplicity table: {multiplicity_output}")
    print(f"near-threshold winner table: {winners_output}")
    print(f"near-threshold report: {report_output}")
    return 0


def _verify_coverage(
    records,
    expected: dict[str, list[str]],
    *,
    processes: int,
    blocks_per_process: int,
) -> None:
    for task_id, candidate_ids in expected.items():
        for candidate_id in candidate_ids:
            screening = [
                row
                for row in records
                if row.phase == "screening"
                and row.task_id == task_id
                and row.candidate_id == candidate_id
            ]
            if len(screening) != blocks_per_process:
                raise RuntimeError(
                    f"{task_id}/{candidate_id} has {len(screening)} screening blocks"
                )
            confirmation = [
                row
                for row in records
                if row.phase == "confirmation"
                and row.task_id == task_id
                and row.candidate_id == candidate_id
            ]
            process_ids = {row.process_id for row in confirmation}
            if len(process_ids) != processes or len(confirmation) != processes * blocks_per_process:
                raise RuntimeError(
                    f"{task_id}/{candidate_id} confirmation coverage is incomplete"
                )
            if any(
                not row.correctness_passed or not row.contract_passed
                for row in screening + confirmation
            ):
                raise RuntimeError(f"{task_id}/{candidate_id} contains an invalid timing block")


def _verify_ledger(campaign: Path) -> None:
    ledger = campaign / "SHA256SUMS"
    if not ledger.exists():
        raise RuntimeError(f"missing checksum ledger: {ledger}")
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = campaign / relative.lstrip("* ")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"checksum mismatch: {path}")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("no winner rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    *,
    state: dict[str, object],
    selected: dict[str, object],
    winners: list[dict[str, object]],
    multiplicity_path: Path,
    environment: dict[str, object],
) -> None:
    with multiplicity_path.open(newline="", encoding="utf-8") as handle:
        multiplicity = list(csv.DictReader(handle))
    lines = [
        "# Workshop 2026 Near-Threshold Multiplicity Stress Test",
        "",
        "Status: complete, checksum-verified, and derived without additional CUDA execution.",
        "",
        f"- Frozen primary candidates: {sum(len(v) for v in selected['tasks'].values())}.",
        f"- Recorded worker hours: {float(state['recorded_worker_hours']):.3f}.",
        f"- GPU: {environment['gpu']}.",
        (
            f"- Software: PyTorch {environment['torch']}; "
            f"Triton {environment['triton']}; Python {environment['python']}."
        ),
        "- Calibration was disjoint from primary timing and excluded from every estimate.",
        "- Every selected candidate was inside the prespecified calibration window.",
        "",
        "## Full-budget frozen winners",
        "",
        "| Task | Candidate | Screen | Confirm | Label |",
        "|---|---|---:|---:|---|",
    ]
    for row in winners:
        lines.append(
            f"| {row['task']} | {row['winner_candidate']} | "
            f"{float(row['screening_speedup']):.4f}x | "
            f"{float(row['confirmation_speedup']):.4f}x | {row['label']} |"
        )
    lines.extend(
        [
            "",
            "## Candidate-budget analysis",
            "",
            "| K | Apparent win rate | Confirmed win rate | Median log optimism | 95% interval |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in multiplicity:
        lines.append(
            f"| {row['candidate_budget']} | {float(row['apparent_win_rate']):.4f} | "
            f"{float(row['confirmed_win_rate']):.4f} | "
            f"{float(row['median_selection_optimism_log']):.6f} | "
            f"[{float(row['selection_optimism_log_ci_lower']):.6f}, "
            f"{float(row['selection_optimism_log_ci_upper']):.6f}] |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_environment(campaign: Path) -> dict[str, object]:
    paths = sorted((campaign / "screening").glob("**/result.json"))
    if not paths:
        raise RuntimeError("screening environment record is missing")
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    environment = data.get("environment") or {}
    devices = environment.get("cuda_devices") or []
    if not devices:
        raise RuntimeError("screening CUDA device record is missing")
    return {
        "gpu": devices[0]["name"],
        "torch": environment["torch_version"],
        "triton": environment["triton_version"],
        "python": str(environment["python_version"]).split()[0],
    }


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
