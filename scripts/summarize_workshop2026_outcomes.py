#!/usr/bin/env python3
"""Derive candidate-failure and compiler-rung summaries from frozen artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


CONTRACT_ERRORS = {
    "alias_contract_mismatch",
    "bad_output_type",
    "input_mutation_mismatch",
    "nondeterministic_output_contract",
    "nondeterministic_output_tree",
    "output_tree_mismatch",
    "special_value_mask_mismatch",
    "unexpected_input_mutation",
    "wrong_dtype",
    "wrong_num_outputs",
    "wrong_shape",
}
CORRECTNESS_ERRORS = {
    "nondeterministic_output_error",
    "nondeterministic_output_values",
    "values_not_close",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign-dir",
        default="artifacts/workshop2026/holdout_campaign",
    )
    parser.add_argument(
        "--failure-output",
        default="reports/tables/workshop2026_candidate_failure_breakdown.csv",
    )
    parser.add_argument(
        "--compiler-output",
        default="reports/tables/workshop2026_compiler_rung.csv",
    )
    parser.add_argument(
        "--report-output",
        default="reports/workshop2026_outcome_audit.md",
    )
    args = parser.parse_args()
    campaign = Path(args.campaign_dir)
    failures = summarize_failures(campaign)
    compiler = summarize_compiler_rung(campaign)
    _write_csv(Path(args.failure_output), failures)
    _write_csv(Path(args.compiler_output), compiler)
    _write_report(Path(args.report_output), failures, compiler)
    print(f"failure breakdown: {args.failure_output}")
    print(f"compiler rung: {args.compiler_output}")
    print(f"outcome audit: {args.report_output}")
    return 0


def summarize_failures(campaign: Path) -> list[dict[str, Any]]:
    counts = {
        "static_policy_failure": 0,
        "contract_failure": 0,
        "correctness_failure": 0,
        "candidate_compile_or_runtime_failure": 0,
        "full_gate_pass": 0,
        "not_evaluated_compiler_baseline_failure": 0,
    }
    task_baseline_failures = 0
    result_paths = sorted((campaign / "screening").glob("*/screening/result.json"))
    for path in result_paths:
        task_result = json.loads(path.read_text(encoding="utf-8"))
        if task_result.get("status") != "completed":
            task_baseline_failures += 1
            candidates = task_result.get("job", {}).get("candidates", [])
            counts["not_evaluated_compiler_baseline_failure"] += len(candidates)
            continue
        for candidate in task_result.get("candidate_results", []):
            policy = candidate.get("policy") or {}
            verification = candidate.get("verification") or {}
            runtime_policy = candidate.get("runtime_policy") or {}
            if policy.get("passed") is not True:
                counts["static_policy_failure"] += 1
                continue
            if candidate.get("status") == "failed":
                counts["candidate_compile_or_runtime_failure"] += 1
                continue
            if verification.get("passed") is not True:
                error_types = {
                    case.get("error_type")
                    for case in verification.get("cases", [])
                    if case.get("error_type")
                }
                if error_types & CONTRACT_ERRORS:
                    counts["contract_failure"] += 1
                elif error_types & CORRECTNESS_ERRORS:
                    counts["correctness_failure"] += 1
                else:
                    counts["candidate_compile_or_runtime_failure"] += 1
                continue
            if runtime_policy.get("passed") is not True or not candidate.get("paired_timing"):
                counts["candidate_compile_or_runtime_failure"] += 1
                continue
            counts["full_gate_pass"] += 1

    evaluated = sum(
        count
        for category, count in counts.items()
        if category != "not_evaluated_compiler_baseline_failure"
    )
    generated = evaluated + counts["not_evaluated_compiler_baseline_failure"]
    if generated != 144 or evaluated != 141 or task_baseline_failures != 1:
        raise RuntimeError(
            "unexpected corrected-campaign funnel: "
            f"generated={generated}, evaluated={evaluated}, "
            f"baseline_failures={task_baseline_failures}"
        )
    notes = {
        "static_policy_failure": "AST policy rejected fallback or module calls",
        "contract_failure": "output tree, shape, dtype, alias, or mutation contract",
        "correctness_failure": "numerical mismatch or nondeterministic values",
        "candidate_compile_or_runtime_failure": "candidate compile, verification runtime, OOM, or CUDA failure",
        "full_gate_pass": "policy, contract, correctness, runtime audit, and timing passed",
        "not_evaluated_compiler_baseline_failure": "one task-level torch.compile OOM blocked all three candidates",
    }
    return [
        {
            "category": category,
            "candidates": count,
            "denominator": generated,
            "notes": notes[category],
        }
        for category, count in counts.items()
    ]


def summarize_compiler_rung(campaign: Path) -> list[dict[str, Any]]:
    frozen = json.loads(
        (campaign / "screening_winners_frozen.json").read_text(encoding="utf-8")
    )
    result_by_task: dict[str, dict[str, Any]] = {}
    completed_baselines = 0
    failed_baselines = 0
    for path in sorted((campaign / "screening").glob("*/screening/result.json")):
        task_result = json.loads(path.read_text(encoding="utf-8"))
        task_id = task_result.get("job", {}).get("task_id")
        result_by_task[task_id] = task_result
        if task_result.get("status") == "completed" and task_result.get("torch_compile"):
            completed_baselines += 1
        else:
            failed_baselines += 1

    winner_rows = []
    for winner in frozen["winners"]:
        task_result = result_by_task[winner["task_id"]]
        candidate = next(
            row
            for row in task_result["candidate_results"]
            if row["candidate_id"] == winner["candidate_id"]
        )
        blocks = candidate["paired_timing"]["blocks"]
        compiler_logs = [
            math.log(
                block["median_ms_per_launch"]["compile"]
                / block["median_ms_per_launch"]["candidate"]
            )
            for block in blocks
        ]
        compiler_speedup = math.exp(statistics.median(compiler_logs))
        winner_rows.append(
            {
                "task_id": winner["task_id"],
                "compiler_speedup": compiler_speedup,
                "eager_speedup": float(winner["screening_speedup"]),
            }
        )

    compile_wins = [row for row in winner_rows if row["compiler_speedup"] > 1.0]
    margin_wins = [row for row in winner_rows if row["compiler_speedup"] > 1.02]
    eager_margin_wins = [row for row in winner_rows if row["eager_speedup"] > 1.02]
    if len(winner_rows) != 10 or len(margin_wins) != 1 or eager_margin_wins:
        raise RuntimeError("unexpected compiler-rung counts in corrected campaign")
    only = margin_wins[0]
    return [
        {
            "scope": "selected_task_baselines",
            "units": 48,
            "compiler_available": completed_baselines,
            "above_compile_parity": "",
            "above_compile_1_02": "",
            "above_eager_1_02": "",
            "notes": f"{failed_baselines} torch.compile baseline OOM; no other unavailable baseline",
        },
        {
            "scope": "frozen_valid_task_winners_screening",
            "units": len(winner_rows),
            "compiler_available": len(winner_rows),
            "above_compile_parity": len(compile_wins),
            "above_compile_1_02": len(margin_wins),
            "above_eager_1_02": len(eager_margin_wins),
            "notes": (
                f"only {only['task_id']} cleared compile margin: "
                f"{only['compiler_speedup']:.6f}x vs compile, "
                f"{only['eager_speedup']:.6f}x vs eager"
            ),
        },
        {
            "scope": "fresh_process_confirmation",
            "units": len(winner_rows),
            "compiler_available": 0,
            "above_compile_parity": "",
            "above_compile_1_02": "",
            "above_eager_1_02": 0,
            "notes": "confirmation remeasured candidate versus eager only; compiler was not rerun",
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    failures: list[dict[str, Any]],
    compiler: list[dict[str, Any]],
) -> None:
    by_category = {row["category"]: row["candidates"] for row in failures}
    screening = next(
        row for row in compiler if row["scope"] == "frozen_valid_task_winners_screening"
    )
    baselines = next(row for row in compiler if row["scope"] == "selected_task_baselines")
    lines = [
        "# Workshop 2026 Outcome Audit",
        "",
        "Derived only from the frozen corrected-campaign records.",
        "",
        "## Candidate funnel",
        "",
    ]
    lines.extend(
        f"- {row['category']}: {row['candidates']}." for row in failures
    )
    lines.extend(
        [
            "",
            "The mutually exclusive evaluated-candidate categories sum to 141. The",
            "three unevaluated candidates belong to one task whose compiler baseline",
            "failed before candidate evaluation.",
            "",
            "## Compiler rung",
            "",
            f"- Compiler baselines materialized for {baselines['compiler_available']}/48 selected tasks.",
            f"- {screening['above_compile_1_02']}/10 frozen valid-task winners cleared the 2% compiler margin at screening.",
            f"- {screening['above_eager_1_02']}/10 cleared the 2% eager margin.",
            "- Confirmation remeasured candidate versus eager only; it did not rerun the compiler.",
            "",
            "These counts do not alter the primary eager-relative promotion result.",
        ]
    )
    if by_category["contract_failure"] == 0:
        lines.append("No candidate failed only an output-contract check in this campaign.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
