from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
REPORT = ROOT / "reports" / "statistical_notes.md"
FUSED8_RECOVERY_REPORT = ROOT / "reports" / "fused8_artifact_recovery_notes.md"
ONE_SHOT_RUN = ROOT / "runs" / "20260520_202314"
REPAIR_RUN = ROOT / "runs" / "20260520_213128"
IMPORT_ROOT = ROOT / "artifacts" / "runpod_imports"


VERIFICATION_COUNTS = [
    {
        "study": "fused8",
        "source": "template",
        "successes": 160,
        "trials": 160,
        "notes": "fixed rigorous deterministic-template budget",
    },
    {
        "study": "fused8",
        "source": "Gemini",
        "successes": 23,
        "trials": 24,
        "notes": "fixed shared fused8 prompt budget",
    },
    {
        "study": "fused8",
        "source": "OpenAI mini",
        "successes": 12,
        "trials": 24,
        "notes": "fixed shared fused8 prompt budget; prompt not separately tuned",
    },
    {
        "study": "Historical KB (affected): pilot",
        "source": "Gemini one-shot",
        "successes": 3,
        "trials": 20,
        "notes": "historical affected-adapter count; not model-accuracy evidence",
    },
    {
        "study": "Historical KB (affected): repair1",
        "source": "Gemini repair",
        "successes": 1,
        "trials": 8,
        "notes": "historical affected-adapter count; not repair-effectiveness evidence",
    },
    {
        "study": "Historical KB (affected): combined",
        "source": "Gemini one-shot + repair1",
        "successes": 4,
        "trials": 20,
        "notes": "historical affected-adapter count; not a current correctness rate",
    },
]


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    one_shot = _load_check(ONE_SHOT_RUN)
    repair = _load_check(REPAIR_RUN)
    one_shot_records = _load_jsonl(ONE_SHOT_RUN / "results.jsonl")
    repair_records = _load_jsonl(REPAIR_RUN / "results.jsonl")

    verification_rows = _verification_rows()
    family_rows = _kernelbench_family_rows(one_shot_records, repair_records)
    flip_rows = _single_run_repeat_flip_rows()
    fused8_recovery_rows = _fused8_uncertainty_recovery_rows()
    memory_rows = _memory_filter_rows(one_shot)
    statistical_rows = _statistical_summary_rows(one_shot, repair, flip_rows)

    _write_csv(TABLES / "verification_rate_intervals.csv", verification_rows)
    _write_csv(TABLES / "kernelbench_family_summary.csv", family_rows)
    _write_csv(TABLES / "single_run_repeat_flip_summary.csv", flip_rows)
    _write_csv(TABLES / "fused8_uncertainty_recovered.csv", fused8_recovery_rows)
    _write_csv(TABLES / "kernelbench_memory_filter_summary.csv", memory_rows)
    _write_csv(TABLES / "statistical_summary.csv", statistical_rows)
    _write_notes(verification_rows, family_rows, flip_rows, memory_rows)
    _write_fused8_recovery_notes(fused8_recovery_rows, flip_rows)

    print(f"Wrote {TABLES / 'verification_rate_intervals.csv'}")
    print(f"Wrote {TABLES / 'kernelbench_family_summary.csv'}")
    print(f"Wrote {TABLES / 'single_run_repeat_flip_summary.csv'}")
    print(f"Wrote {TABLES / 'fused8_uncertainty_recovered.csv'}")
    print(f"Wrote {TABLES / 'kernelbench_memory_filter_summary.csv'}")
    print(f"Wrote {TABLES / 'statistical_summary.csv'}")
    print(f"Wrote {REPORT}")
    print(f"Wrote {FUSED8_RECOVERY_REPORT}")
    return 0


def _load_check(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "kernelbench_l1_check.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    if trials <= 0:
        return 0.0, 0.0, 0.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _fmt_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _verification_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in VERIFICATION_COUNTS:
        rate, lo, hi = _wilson(int(item["successes"]), int(item["trials"]))
        rows.append(
            {
                "study": item["study"],
                "source": item["source"],
                "successes": item["successes"],
                "trials": item["trials"],
                "rate": _fmt_float(rate),
                "wilson_lo": _fmt_float(lo),
                "wilson_hi": _fmt_float(hi),
                "notes": item["notes"],
                "evidence_status": (
                    "supported_fused8_observation"
                    if item["study"] == "fused8"
                    else "historical_adapter_output_only"
                ),
            }
        )
    return rows


def _kernelbench_family_rows(
    one_shot_records: list[dict[str, Any]],
    repair_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    families = ["convolution", "matmul", "pooling", "loss"]
    selected = Counter(str(r.get("op_family")) for r in one_shot_records)
    one_verified = Counter(str(r.get("op_family")) for r in one_shot_records if r.get("verification_passed"))
    one_stable = Counter(
        str(r.get("op_family"))
        for r in one_shot_records
        if r.get("candidate_label") == "REPEAT_STABLE_WIN"
    )
    repair_attempted = Counter(str(r.get("op_family")) for r in repair_records)
    repair_verified = Counter(str(r.get("op_family")) for r in repair_records if r.get("verification_passed"))
    repair_stable = Counter(
        str(r.get("op_family"))
        for r in repair_records
        if r.get("candidate_label") == "REPEAT_STABLE_WIN"
    )
    combined_correct: dict[str, set[str]] = defaultdict(set)
    combined_stable: dict[str, set[str]] = defaultdict(set)
    for record in one_shot_records + repair_records:
        family = str(record.get("op_family"))
        task = str(record.get("task_id"))
        if record.get("verification_passed"):
            combined_correct[family].add(task)
        if record.get("candidate_label") == "REPEAT_STABLE_WIN":
            combined_stable[family].add(task)

    rows = []
    for family in families:
        rows.append(
            {
                "family": family,
                "selected_tasks": selected[family],
                "one_shot_verified": one_verified[family],
                "one_shot_stable_wins": one_stable[family],
                "repair_attempted": repair_attempted[family],
                "repair_verified": repair_verified[family],
                "repair_stable_wins": repair_stable[family],
                "combined_unique_correct": len(combined_correct[family]),
                "combined_stable_wins": len(combined_stable[family]),
                "notes": _family_note(family, one_verified[family], one_stable[family], repair_verified[family]),
                "evidence_status": "historical_adapter_output_only",
            }
        )
    return rows


def _family_note(family: str, one_verified: int, one_stable: int, repair_verified: int) -> str:
    if family == "loss":
        return "affected evaluator recorded all stable labels in this family"
    if one_verified or repair_verified:
        return "affected evaluator recorded one verification and no stable label"
    return "affected evaluator recorded no verification"


def _single_run_repeat_flip_rows() -> list[dict[str, Any]]:
    path = TABLES / "fused8_template_results.csv"
    if not path.exists():
        return [
            {
                "scope": "fused8 template all 160 candidates",
                "artifact_source": "runs/20260520_155839",
                "single_run_above_eager": "not preserved",
                "repeat_below_eager": "not preserved",
                "flip_rate": "not preserved",
                "notes": "full per-candidate fused8 artifacts are missing locally",
            }
        ]

    with path.open(newline="", encoding="utf-8") as f:
        task_rows = list(csv.DictReader(f))
    above = [
        row
        for row in task_rows
        if _to_float(row.get("best_single_run_speedup_vs_eager")) is not None
        and _to_float(row.get("best_single_run_speedup_vs_eager")) > 1.0
    ]
    flips = [
        row
        for row in above
        if _to_float(row.get("repeat_median_speedup_vs_eager")) is not None
        and _to_float(row.get("repeat_median_speedup_vs_eager")) < 1.0
    ]
    rows = [
        {
            "scope": "fused8 template task-best summary",
            "artifact_source": "reports/tables/fused8_template_results.csv",
            "single_run_above_eager": len(above),
            "repeat_below_eager": len(flips),
            "flip_rate": _fmt_float(len(flips) / len(above)) if above else "not applicable",
            "notes": "task-best imported summary only; bias_relu is the observed flip",
        },
        {
            "scope": "fused8 template all 160 candidates",
            "artifact_source": "runs/20260520_155839",
            "single_run_above_eager": "not preserved",
            "repeat_below_eager": "not preserved",
            "flip_rate": "not preserved",
            "notes": "full per-candidate single-run and repeat-median pairs are not preserved locally",
        },
    ]
    return rows


def _fused8_uncertainty_recovery_rows() -> list[dict[str, Any]]:
    sources = [
        ("template", "20260520_155839", "160/160"),
        ("Gemini", "20260520_163344", "23/24"),
        ("OpenAI mini", "20260520_163607", "12/24"),
    ]
    rows: list[dict[str, Any]] = []
    for source_name, run_id, verified in sources:
        run_dir = _find_run_dir(run_id)
        recovered = _scan_fused8_run(run_dir) if run_dir else {}
        rows.append(
            {
                "source": source_name,
                "run_id": run_id,
                "verified": verified,
                "artifact_path": str(run_dir.relative_to(ROOT)) if run_dir and run_dir.is_relative_to(ROOT) else ("missing" if not run_dir else str(run_dir)),
                "repeat_medians": "yes" if _has_imported_medians(source_name) else "not preserved",
                "p25_p75_iqr": "yes" if recovered.get("iqr") else "not preserved",
                "bootstrap_ci": "yes" if recovered.get("bootstrap_ci") else "not preserved",
                "std_cv": "yes" if recovered.get("std_cv") else _std_cv_from_imported_rows(source_name),
                "per_session_medians": "yes" if recovered.get("per_session_medians") else "not preserved",
                "per_candidate_flip_pairs": "yes" if recovered.get("per_candidate_flip_pairs") else "not preserved",
                "notes": _fused8_recovery_note(run_id, run_dir, recovered),
            }
        )
    return rows


def _find_run_dir(run_id: str) -> Path | None:
    candidates = [
        ROOT / "runs" / run_id,
        IMPORT_ROOT / "runs" / run_id,
    ]
    for path in candidates:
        if path.exists():
            return path
    if (ROOT / "artifacts").exists():
        for path in (ROOT / "artifacts").rglob(run_id):
            if path.is_dir():
                return path
    return None


def _scan_fused8_run(run_dir: Path | None) -> dict[str, bool]:
    found = {
        "iqr": False,
        "bootstrap_ci": False,
        "std_cv": False,
        "per_session_medians": False,
        "per_candidate_flip_pairs": False,
    }
    if run_dir is None:
        return found
    candidate_files = []
    for rel in ["results.jsonl", "repeatability_results.json", "fused8_report.md", "repeatability_report.md"]:
        path = run_dir / rel
        if path.exists():
            candidate_files.append(path)
    for path in candidate_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        found["iqr"] = found["iqr"] or "iqr" in lowered or "p25" in lowered or "p75" in lowered
        found["bootstrap_ci"] = found["bootstrap_ci"] or "bootstrap_ci" in lowered or "ci_low" in lowered
        found["std_cv"] = found["std_cv"] or "std_ms" in lowered or '"cv"' in lowered or " cv " in lowered
        found["per_session_medians"] = found["per_session_medians"] or "session_summaries" in lowered or "session median" in lowered
        found["per_candidate_flip_pairs"] = found["per_candidate_flip_pairs"] or (
            "single_run" in lowered and "repeat_median" in lowered
        )
    return found


def _has_imported_medians(source_name: str) -> bool:
    path = TABLES / "fused8_model_comparison.csv"
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("baseline") == source_name or (source_name == "template" and row.get("baseline") == "template"):
                return _to_float(row.get("median_speedup_vs_eager")) is not None
    return False


def _std_cv_from_imported_rows(source_name: str) -> str:
    path = TABLES / "fused8_stable_winners.csv"
    if not path.exists() or source_name == "template":
        return "not preserved"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if source_name == "Gemini":
        hits = [row for row in rows if row.get("uncertainty", "").startswith("std") and row.get("source_type") in {"n/a", "llm"}]
    else:
        hits = [row for row in rows if row.get("stable_winner") == "OpenAI mini" and row.get("uncertainty", "").startswith("std")]
    return "partially preserved" if hits else "not preserved"


def _fused8_recovery_note(run_id: str, run_dir: Path | None, recovered: dict[str, bool]) -> str:
    if run_dir is None:
        return "full run directory is missing locally and under artifacts/runpod_imports"
    fields = [name for name, value in recovered.items() if value]
    if fields:
        return "recovered fields: " + ", ".join(fields)
    return "run directory present but no full interval or per-candidate flip fields detected"


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).replace("x", ""))
    except (TypeError, ValueError):
        return None


def _memory_filter_rows(one_shot: dict[str, Any]) -> list[dict[str, Any]]:
    records = one_shot.get("records") or []
    skipped = [record for record in records if record.get("skipped")]
    selected = [record for record in records if not record.get("skipped")]
    skipped_by_family = Counter(str(record.get("op_family")) for record in skipped)
    selected_by_family = Counter(str(record.get("op_family")) for record in selected)
    families = sorted(set(skipped_by_family) | set(selected_by_family))
    rows: list[dict[str, Any]] = [
        {
            "category": "overall",
            "family": "all",
            "count": one_shot.get("tasks_loaded", "not preserved"),
            "shape_examples": "not applicable",
            "interpretation": "historical adapter loaded official L1 task pool",
            "evidence_status": "historical_adapter_output_only",
        },
        {
            "category": "selected_feasible",
            "family": "all",
            "count": len(selected),
            "shape_examples": "see selected task appendix",
            "interpretation": "historical selection after lower-bound memory filtering",
            "evidence_status": "historical_adapter_output_only",
        },
        {
            "category": "skipped_memory_cap",
            "family": "all",
            "count": len(skipped),
            "shape_examples": _shape_examples(skipped),
            "interpretation": "historical lower-bound filter skip encountered before selection cap",
            "evidence_status": "historical_adapter_output_only",
        },
    ]
    for family in families:
        rows.append(
            {
                "category": "selected_feasible",
                "family": family,
                "count": selected_by_family[family],
                "shape_examples": _shape_examples([r for r in selected if str(r.get("op_family")) == family]),
                "interpretation": "historical family count after feasible selection",
                "evidence_status": "historical_adapter_output_only",
            }
        )
        rows.append(
            {
                "category": "skipped_memory_cap",
                "family": family,
                "count": skipped_by_family[family],
                "shape_examples": _shape_examples([r for r in skipped if str(r.get("op_family")) == family]),
                "interpretation": "historical lower-bound filter count by family in scanned prefix",
                "evidence_status": "historical_adapter_output_only",
            }
        )
    return rows


def _shape_examples(records: list[dict[str, Any]], limit: int = 2) -> str:
    examples = []
    for record in records:
        shape = record.get("shape")
        if shape is not None:
            examples.append(str(shape).replace(" ", ""))
        if len(examples) >= limit:
            break
    return "; ".join(examples) if examples else "not preserved"


def _statistical_summary_rows(
    one_shot: dict[str, Any],
    repair: dict[str, Any],
    flip_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "metric": "stable-win count caveat",
            "value": "small counts",
            "artifact_source": "reports/tables/fused8_model_comparison.csv and reports/tables/kernelbench_l1_pilot.csv",
            "notes": "fused8 counts are descriptive; KernelBench labels are historical affected-adapter output",
        },
        {
            "metric": "independent sessions",
            "value": "3",
            "artifact_source": "KernelBench timing metadata and rigorous fused8 configs",
            "notes": "practical stability check; not high-powered across-session variance estimation",
        },
        {
            "metric": "KernelBench tasks loaded",
            "value": str(one_shot.get("tasks_loaded", "not preserved")),
            "artifact_source": "runs/20260520_202314/kernelbench_l1_check.json",
            "notes": "historical adapter field; not current external validation",
        },
        {
            "metric": "KernelBench memory skips encountered before selection cap",
            "value": str(one_shot.get("tasks_skipped", "not preserved")),
            "artifact_source": "runs/20260520_202314/kernelbench_l1_check.json",
            "notes": "historical incomplete lower-bound filter; not current external validation",
        },
        {
            "metric": "fused8 all-candidate flip frequency",
            "value": "not preserved",
            "artifact_source": "runs/20260520_155839 missing locally",
            "notes": str(flip_rows[-1].get("notes")),
        },
    ]


def _write_notes(
    verification_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    flip_rows: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Statistical Notes From Existing Artifacts",
        "",
        "This analysis uses only existing CSV and run artifacts. It does not run experiments, call model APIs, or relabel results. KernelBench rows come from the affected historical adapter and are audit metadata, not model-accuracy or performance evidence.",
        "",
        "## Verification-Rate Intervals",
        "",
        "Verification rates use 95% Wilson intervals under the fixed candidate budgets.",
        "",
        "| Study | Source | Successes | Trials | Rate | Wilson 95% CI |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in verification_rows:
        lines.append(
            f"| {row['study']} | {row['source']} | {row['successes']} | {row['trials']} | "
            f"{row['rate']} | [{row['wilson_lo']}, {row['wilson_hi']}] |"
        )
    lines.extend(
        [
            "",
            "Candidate-level model significance testing is omitted because multiple candidates share each fused8 task and are not independent Bernoulli trials.",
            "",
            "## KernelBench Family Summary",
            "",
            "The affected historical evaluator recorded nonuniform outcomes by family. The table documents its behavior; it is not a corrected-adapter family estimate.",
            "",
            "| Family | Selected | One-shot verified | One-shot stable | Repair attempted | Repair verified | Combined correct |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in family_rows:
        lines.append(
            f"| {row['family']} | {row['selected_tasks']} | {row['one_shot_verified']} | "
            f"{row['one_shot_stable_wins']} | {row['repair_attempted']} | {row['repair_verified']} | "
            f"{row['combined_unique_correct']} |"
        )
    lines.extend(
        [
            "",
            "## Single-Run Versus Repeat Flips",
            "",
        ]
    )
    for row in flip_rows:
        lines.append(
            f"- {row['scope']}: single-run above eager = {row['single_run_above_eager']}, "
            f"repeat below eager = {row['repeat_below_eager']}, flip rate = {row['flip_rate']}. "
            f"{row['notes']}"
        )
    lines.extend(
        [
            "",
            "## Memory Filtering",
            "",
            "Historical KernelBench memory filtering is characterized in `reports/tables/kernelbench_memory_filter_summary.csv`. "
            "It was a lower-bound, deterministic selector and is retained as audit provenance.",
            "",
            "## Caveats",
            "",
            "- Fused8 stable-win counts are small and should not be read as statistically significant rankings.",
            "- KernelBench verification, family, and stable-label counts are provisional because the source adapter was invalid.",
            "- Candidate search creates multiplicity; repeatability labels reduce but do not eliminate selection bias.",
            "- Three independent sessions are a practical stability check, not a high-powered variance estimate.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_fused8_recovery_notes(
    recovery_rows: list[dict[str, Any]],
    flip_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Fused8 Artifact Recovery Notes",
        "",
        "This recovery pass searches the local workspace and `artifacts/runpod_imports` for the rigorous fused8 run directories. It does not run benchmarks or infer missing interval statistics.",
        "",
        "## Recovery Status",
        "",
        "| Source | Run | Artifact path | IQR | Bootstrap CI | Std/CV | Per-session medians | Flip pairs | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in recovery_rows:
        lines.append(
            f"| {row['source']} | {row['run_id']} | `{row['artifact_path']}` | {row['p25_p75_iqr']} | "
            f"{row['bootstrap_ci']} | {row['std_cv']} | {row['per_session_medians']} | "
            f"{row['per_candidate_flip_pairs']} | {row['notes']} |"
        )
    lines.extend(["", "## Flip-Frequency Status", ""])
    for row in flip_rows:
        lines.append(
            f"- {row['scope']}: `{row['flip_rate']}` ({row['notes']})"
        )
    lines.extend(
        [
            "",
            "The local package therefore retains the task-best `bias_relu` flip but not the full 160-candidate single-run-to-repeat flip frequency. If the missing RunPod fused8 directories are later imported, rerun `python scripts/analyze_existing_results_statistics.py` and rebuild paper assets.",
        ]
    )
    FUSED8_RECOVERY_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
