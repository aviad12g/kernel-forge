"""Fused8 repeatability comparison and curated dataset export."""

from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from openkernelforge.reports.run_data import load_run_bundle, read_artifact


FUSED8_TASKS = [
    "bias_relu",
    "sigmoid_mul",
    "add_relu",
    "residual_add_relu",
    "bias_gelu",
    "row_sum",
    "layernorm_small",
    "rmsnorm_small",
]

CURATED_FILES = [
    "correct_fast_repeat_stable.jsonl",
    "correct_fast_single_run.jsonl",
    "correct_promising.jsonl",
    "optimization_pairs_template_vs_gemini.jsonl",
    "optimization_pairs_gemini_vs_template.jsonl",
    "rejected_or_unstable.jsonl",
]


def curate_fused8_dataset(
    *,
    template_run: str | Path,
    gemini_run: str | Path,
    template_guided_run: str | Path,
    out_dir: str | Path,
) -> Path:
    """Create a curated fused8 dataset from template and Gemini runs."""

    runs = _load_named_runs(template_run, gemini_run, template_guided_run)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rows = {filename: [] for filename in CURATED_FILES}
    all_candidate_rows: list[dict[str, Any]] = []

    for label, data in runs.items():
        repeatability = _load_repeatability_index(data["run_dir"])
        for record in data["bundle"]["candidate_records"]:
            row = _candidate_row(label, data["run_dir"], record, repeatability)
            if not row:
                continue
            all_candidate_rows.append(row)
            target_file = _candidate_split(row)
            rows[target_file].append(row)

    pair_rows = _optimization_pair_rows(all_candidate_rows)
    rows["optimization_pairs_template_vs_gemini.jsonl"].extend(
        row for row in pair_rows if row["target_type"] == "template_vs_gemini"
    )
    rows["optimization_pairs_gemini_vs_template.jsonl"].extend(
        row for row in pair_rows if row["target_type"] == "gemini_vs_template"
    )
    _deduplicate_stable_fast_rows(rows)

    counts: dict[str, int] = {}
    for filename in CURATED_FILES:
        _write_jsonl(out_path / filename, rows[filename])
        counts[filename] = len(rows[filename])

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_runs": {
            "template": str(Path(template_run)),
            "gemini": str(Path(gemini_run)),
            "gemini_template_guided": str(Path(template_guided_run)),
        },
        "counts_by_file": counts,
        "counts_by_label": dict(sorted(Counter(row["label"] for row in all_candidate_rows).items())),
        "task_split": _task_split(rows),
        "repeatability_status": {
            label: _repeatability_status(data["run_dir"]) for label, data in runs.items()
        },
        "warnings": [
            "Internal fused8 benchmark only; not KernelBench and not a SOTA claim.",
            "Do not use single-run-only or unstable rows as direct SFT targets without review.",
        ],
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_path / "README.md").write_text(_readme(manifest), encoding="utf-8")

    write_fused8_repeatability_comparison(
        template_run=template_run,
        gemini_run=gemini_run,
        template_guided_run=template_guided_run,
        out=Path("runs") / "fused8_repeatability_comparison.md",
    )
    write_fused8_phase11_conclusion(
        template_run=template_run,
        gemini_run=gemini_run,
        template_guided_run=template_guided_run,
        dataset_dir=out_path,
        out=Path("runs") / "fused8_phase11_conclusion.md",
    )
    return out_path


def inspect_curated_fused8_dataset(dataset_dir: str | Path) -> Path:
    """Write a human-readable inspection report for a curated fused8 dataset."""

    root = Path(dataset_dir)
    rows_by_file = _load_curated_rows(root)
    manifest = _load_manifest(root)
    all_rows = [row for rows in rows_by_file.values() for row in rows]
    stable = rows_by_file.get("correct_fast_repeat_stable.jsonl", [])
    optimization = (
        rows_by_file.get("optimization_pairs_template_vs_gemini.jsonl", [])
        + rows_by_file.get("optimization_pairs_gemini_vs_template.jsonl", [])
    )
    red_flags = _curated_red_flags(rows_by_file)

    lines = [
        "# Curated Fused8 Dataset Inspection",
        "",
        "Internal OpenKernelForge fused8 dataset. Do not train on this before manual review.",
        "",
        f"- Dataset dir: `{root}`",
        f"- Source runs: `{json.dumps(manifest.get('source_runs', {}), sort_keys=True)}`",
        "",
        "## Counts Per Split",
        "",
    ]
    for filename in CURATED_FILES:
        lines.append(f"- {filename}: {len(rows_by_file.get(filename, []))}")

    lines.extend(["", "## Counts Per Task", ""])
    for task_id, count in sorted(Counter(row.get("task_id") for row in all_rows).items()):
        lines.append(f"- {task_id}: {count}")

    lines.extend(["", "## Counts Per Source Type", ""])
    for source, count in sorted(Counter(row.get("source_type") for row in all_rows).items()):
        lines.append(f"- {source}: {count}")

    lines.extend(
        [
            "",
            "## Stable-Fast Rows",
            "",
            "| Task | Source | Repeat median | Single-run speedup | Candidate | Source summary |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in stable:
        repeat = row.get("repeatability") or {}
        stats = repeat.get("stats") or {}
        lines.append(
            "| {task} | {source} | {repeat} | {single} | `{path}` | {summary} |".format(
                task=row.get("task_id"),
                source=row.get("source_type"),
                repeat=_fmt(stats.get("median")),
                single=_fmt(row.get("speedup_vs_eager")),
                path=row.get("candidate_path"),
                summary=_source_summary(row.get("candidate_code") or ""),
            )
        )

    lines.extend(
        [
            "",
            "## Optimization Pairs",
            "",
            "| Task | Slower source | Faster source | Speedup delta | Faster repeat-stable | Faster path |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for row in optimization:
        fast_repeat = row.get("fast_repeatability") or {}
        fast_stats = fast_repeat.get("stats") or {}
        lines.append(
            "| {task} | {slow} | {fast} | {delta} | {stable} | `{path}` |".format(
                task=row.get("task_id"),
                slow=row.get("slow_source_type"),
                fast=row.get("fast_source_type"),
                delta=_fmt(row.get("speedup_delta")),
                stable="yes" if fast_repeat.get("stable") and (fast_stats.get("median") or 0) >= 1.0 else "no",
                path=row.get("fast_candidate_path"),
            )
        )

    lines.extend(["", "## Red Flags", ""])
    if red_flags:
        for flag in red_flags:
            lines.append(f"- {flag}")
    else:
        lines.append("- none detected")

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Good for SFT: repeat-stable fast rows after manual source review.",
            "- Good for optimization training: template-vs-Gemini pairs with repeat-stable faster targets.",
            "- Do not train on this yet; first compare at least one stronger model on the exact fused8 protocol.",
            "",
        ]
    )
    path = root / "curation_inspection_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def validate_curated_fused8_dataset(dataset_dir: str | Path) -> tuple[bool, Path, list[str]]:
    """Validate a curated fused8 dataset and write a report."""

    root = Path(dataset_dir)
    errors: list[str] = []
    rows_by_file: dict[str, list[dict[str, Any]]] = {}
    manifest = _load_manifest(root, errors)
    for filename in CURATED_FILES:
        path = root / filename
        if not path.exists():
            errors.append(f"missing {filename}")
            rows_by_file[filename] = []
            continue
        rows_by_file[filename] = _read_jsonl_checked(path, errors)

    for filename, rows in rows_by_file.items():
        for line_no, row in enumerate(rows, start=1):
            _validate_curated_row(filename, line_no, row, errors)

    for row in rows_by_file.get("correct_fast_repeat_stable.jsonl", []):
        repeat_median = ((row.get("repeatability") or {}).get("stats") or {}).get("median")
        if repeat_median is None or float(repeat_median) < 1.0:
            errors.append(
                f"stable-fast row below repeat median 1.0: {row.get('task_id')} {row.get('candidate_path')}"
            )
    for row in rows_by_file.get("correct_fast_single_run.jsonl", []):
        repeat_median = ((row.get("repeatability") or {}).get("stats") or {}).get("median")
        if repeat_median is not None and float(repeat_median) >= 1.0 and (row.get("repeatability") or {}).get("stable"):
            errors.append(f"stable row mixed into single-run split: {row.get('candidate_path')}")

    stable_hashes = Counter(
        _code_hash(row.get("candidate_code") or "")
        for row in rows_by_file.get("correct_fast_repeat_stable.jsonl", [])
        if row.get("candidate_code")
    )
    duplicates = [hash_value for hash_value, count in stable_hashes.items() if count > 1]
    if duplicates:
        errors.append(f"duplicate candidate code hashes in stable-fast split: {len(duplicates)}")

    manifest_counts = manifest.get("counts_by_file") if isinstance(manifest, dict) else {}
    if isinstance(manifest_counts, dict):
        for filename, rows in rows_by_file.items():
            if manifest_counts.get(filename) != len(rows):
                errors.append(
                    f"manifest count mismatch for {filename}: manifest={manifest_counts.get(filename)} observed={len(rows)}"
                )
    else:
        errors.append("manifest missing counts_by_file")

    lines = [
        "# Curated Fused8 Validation Report",
        "",
        f"- Dataset dir: `{root}`",
        f"- Status: {'PASS' if not errors else 'FAIL'}",
        "",
        "## Counts",
        "",
    ]
    for filename in CURATED_FILES:
        lines.append(f"- {filename}: {len(rows_by_file.get(filename, []))}")
    lines.extend(["", "## Errors", ""])
    if errors:
        for error in errors:
            lines.append(f"- {error}")
    else:
        lines.append("- none")
    report_path = root / "curated_validation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return not errors, report_path, errors


def write_fused8_repeatability_comparison(
    *,
    template_run: str | Path,
    gemini_run: str | Path,
    template_guided_run: str | Path,
    out: str | Path = "runs/fused8_repeatability_comparison.md",
) -> Path:
    """Write a cross-run fused8 repeatability comparison."""

    runs = _load_named_runs(template_run, gemini_run, template_guided_run)
    best_rows = _best_rows_by_task_and_source(runs)
    all_rows = _all_candidate_rows_from_runs(runs)
    lines = [
        "# Fused8 Repeatability Comparison",
        "",
        "Internal OpenKernelForge fused8 benchmark only. This is not KernelBench and not a SOTA claim.",
        "",
        "| Task | Source | Single-run speedup | Repeat median | Repeat std | CV | Stable | Status | Candidate |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    final_winners = _final_stable_winners_from_rows(all_rows)
    for task_id in FUSED8_TASKS:
        for source in ("template", "gemini", "gemini_template_guided"):
            row = best_rows.get((task_id, source))
            if not row:
                lines.append(f"| {task_id} | {source} | n/a | n/a | n/a | n/a | n/a | BELOW_EAGER | n/a |")
                continue
            repeat = row.get("repeatability") or {}
            stats = repeat.get("stats") or {}
            lines.append(
                "| {task} | {source} | {single} | {repeat_median} | {std} | {cv} | {stable} | {status} | `{path}` |".format(
                    task=task_id,
                    source=source,
                    single=_fmt(row.get("speedup_vs_eager")),
                    repeat_median=_fmt(stats.get("median")),
                    std=_fmt(stats.get("std")),
                    cv=_fmt(stats.get("coefficient_of_variation")),
                    stable="yes" if repeat.get("stable") else ("no" if repeat else "missing"),
                    status=_stability_label(row),
                    path=row.get("candidate_path") or "n/a",
                )
            )

    lines.extend(["", "## Final Stable Winners", ""])
    for task_id in FUSED8_TASKS:
        winner = final_winners.get(task_id)
        if winner:
            repeat = winner.get("repeatability") or {}
            lines.append(
                f"- {task_id}: {winner['source_type']} `{winner.get('candidate_path')}` "
                f"repeat median {_fmt((repeat.get('stats') or {}).get('median'))}x"
            )
        else:
            lines.append(f"- {task_id}: no repeat-stable above-eager winner")
    lines.append("")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_fused8_phase11_conclusion(
    *,
    template_run: str | Path,
    gemini_run: str | Path,
    template_guided_run: str | Path,
    dataset_dir: str | Path,
    out: str | Path = "runs/fused8_phase11_conclusion.md",
) -> Path:
    """Write the final fused8 conclusion report."""

    runs = _load_named_runs(template_run, gemini_run, template_guided_run)
    summaries = {label: _run_summary(data["bundle"]) for label, data in runs.items()}
    best_rows = _best_rows_by_task_and_source(runs)
    winners = _final_stable_winners_from_rows(_all_candidate_rows_from_runs(runs))
    manifest_path = Path(dataset_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    lines = [
        "# OpenKernelForge Fused8 Conclusion",
        "",
        "This is an internal fused8 benchmark only. It is not KernelBench and it is not a SOTA claim.",
        "Repeatability matters: single-run speedups are treated as candidates for review, not final wins.",
        "Deterministic templates remain the strongest overall floor unless repeatability proves otherwise.",
        "",
        "## Run Summary",
        "",
        "| Run | Candidates | Verified | Median speedup vs eager | Median speedup vs torch.compile | Tasks > eager | Tasks > compile |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, summary in summaries.items():
        lines.append(
            f"| {label} | {summary['candidates']} | {summary['verified']} | "
            f"{_fmt(summary['median_speedup'])} | {_fmt(summary['median_compile_speedup'])} | "
            f"{summary['tasks_gt_eager']} | {summary['tasks_gt_compile']} |"
        )

    lines.extend(["", "## Final Stable Winner Per Task", ""])
    for task_id in FUSED8_TASKS:
        winner = winners.get(task_id)
        if not winner:
            lines.append(f"- {task_id}: no repeat-stable above-eager winner")
            continue
        repeat = winner.get("repeatability") or {}
        lines.append(
            f"- {task_id}: {winner['source_type']} "
            f"repeat median {_fmt((repeat.get('stats') or {}).get('median'))}x, "
            f"single-run {_fmt(winner.get('speedup_vs_eager'))}x, "
            f"path `{winner.get('candidate_path')}`"
        )

    competitive = _gemini_competitive_tasks(best_rows)
    template_dominated = _template_dominated_tasks(best_rows)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Gemini is competitive on: " + (", ".join(competitive) if competitive else "none in this repeatability view"),
            "- Deterministic templates dominate: " + (", ".join(template_dominated) if template_dominated else "none"),
            "- Good future training targets: stable template winners plus Gemini candidates that are close but slower.",
            "- Deprioritize standalone row_sum/layernorm-style cases if they stay below eager after repeatability.",
            "",
            "## Curated Dataset Counts",
            "",
        ]
    )
    counts = manifest.get("counts_by_file") or {}
    for filename in CURATED_FILES:
        lines.append(f"- {filename}: {counts.get(filename, 0)}")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Use the curated fused8 dataset for manual review and prompt/model comparison. "
            "The next useful step is a stronger model comparison on fused8 and curation of "
            "template-vs-Gemini optimization pairs before any LoRA fine-tuning.",
            "",
        ]
    )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _load_named_runs(
    template_run: str | Path,
    gemini_run: str | Path,
    template_guided_run: str | Path,
) -> dict[str, dict[str, Any]]:
    return {
        "template": {"run_dir": Path(template_run), "bundle": load_run_bundle(template_run)},
        "gemini": {"run_dir": Path(gemini_run), "bundle": load_run_bundle(gemini_run)},
        "gemini_template_guided": {
            "run_dir": Path(template_guided_run),
            "bundle": load_run_bundle(template_guided_run),
        },
    }


def _candidate_row(
    source_label: str,
    run_dir: Path,
    record: dict[str, Any],
    repeatability: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    benchmark = record.get("benchmark_summary") or {}
    speedup = benchmark.get("speedup_vs_eager")
    if speedup is None or not record.get("candidate_path"):
        return None
    task_id = str(record.get("task_id") or "")
    repeat = _match_repeatability(record, repeatability)
    source_type = {
        "template": "template",
        "gemini": "gemini",
        "gemini_template_guided": "gemini_template_guided",
    }[source_label]
    label = _candidate_label(float(speedup), repeat)
    return {
        "task_id": task_id,
        "task_family": record.get("task_family") or "fused8",
        "source_type": source_type,
        "generation_stage": record.get("generation_stage"),
        "candidate_code": read_artifact(record.get("candidate_path"), run_dir=run_dir),
        "candidate_path": record.get("candidate_path"),
        "source_run_dir": str(run_dir),
        "benchmark": benchmark,
        "repeatability": repeat,
        "template_metadata": _template_metadata(record),
        "model": record.get("model"),
        "backend": record.get("backend"),
        "prompt_path": record.get("prompt_path"),
        "response_path": record.get("response_path"),
        "label": label,
        "speedup_vs_eager": float(speedup),
        "speedup_vs_torch_compile": benchmark.get("speedup_vs_torch_compile"),
        "verification_passed": record.get("verification_passed"),
        "policy_passed": record.get("policy_passed"),
    }


def _candidate_split(row: dict[str, Any]) -> str:
    if not (row.get("policy_passed") and row.get("verification_passed")):
        row["label"] = "unstable"
        return "rejected_or_unstable.jsonl"
    label = row["label"]
    if label == "stable_fast":
        return "correct_fast_repeat_stable.jsonl"
    if label == "single_run_fast":
        return "correct_fast_single_run.jsonl"
    if label == "promising":
        return "correct_promising.jsonl"
    return "rejected_or_unstable.jsonl"


def _candidate_label(speedup: float, repeat: dict[str, Any]) -> str:
    stats = repeat.get("stats") or {}
    repeat_median = stats.get("median")
    if repeat_median is not None:
        if repeat.get("stable") and float(repeat_median) >= 1.0:
            return "stable_fast"
        if speedup >= 1.0:
            return "unstable"
    if speedup >= 1.0:
        return "single_run_fast"
    if speedup >= 0.8:
        return "promising"
    return "unstable"


def _optimization_pair_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        if row.get("verification_passed") and row.get("policy_passed"):
            by_task[row["task_id"]].append(row)
    pairs: list[dict[str, Any]] = []
    for task_id, rows in by_task.items():
        template_rows = [row for row in rows if row["source_type"] == "template"]
        gemini_rows = [row for row in rows if row["source_type"] in {"gemini", "gemini_template_guided"}]
        if not template_rows or not gemini_rows:
            continue
        best_template = max(template_rows, key=_effective_speedup)
        best_gemini = max(gemini_rows, key=_effective_speedup)
        if _effective_speedup(best_template) > _effective_speedup(best_gemini):
            pairs.append(_pair_row(task_id, slower=best_gemini, faster=best_template, target_type="template_vs_gemini"))
        elif _effective_speedup(best_gemini) > _effective_speedup(best_template):
            pairs.append(_pair_row(task_id, slower=best_template, faster=best_gemini, target_type="gemini_vs_template"))
    return pairs


def _deduplicate_stable_fast_rows(rows: dict[str, list[dict[str, Any]]]) -> None:
    stable = rows.get("correct_fast_repeat_stable.jsonl", [])
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stable:
        code = row.get("candidate_code") or ""
        by_hash[_code_hash(code)].append(row)
    kept: list[dict[str, Any]] = []
    rejected = rows.setdefault("rejected_or_unstable.jsonl", [])
    for duplicates in by_hash.values():
        best = max(duplicates, key=_effective_speedup)
        kept.append(best)
        for row in duplicates:
            if row is best:
                continue
            duplicate = dict(row)
            duplicate["label"] = "duplicate_stable_fast"
            duplicate["duplicate_of_candidate_path"] = best.get("candidate_path")
            rejected.append(duplicate)
    rows["correct_fast_repeat_stable.jsonl"] = sorted(
        kept,
        key=lambda row: (str(row.get("task_id")), str(row.get("source_type")), str(row.get("candidate_path"))),
    )


def _pair_row(task_id: str, *, slower: dict[str, Any], faster: dict[str, Any], target_type: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "task_family": "fused8",
        "target_type": target_type,
        "label": "optimization_pair",
        "source_type": "template_gemini_comparison",
        "slow_source_type": slower["source_type"],
        "fast_source_type": faster["source_type"],
        "slow_candidate_path": slower.get("candidate_path"),
        "fast_candidate_path": faster.get("candidate_path"),
        "slow_code": slower.get("candidate_code"),
        "fast_code": faster.get("candidate_code"),
        "target": faster.get("candidate_code"),
        "slow_benchmark": slower.get("benchmark"),
        "fast_benchmark": faster.get("benchmark"),
        "slow_repeatability": slower.get("repeatability"),
        "fast_repeatability": faster.get("repeatability"),
        "speedup_delta": _effective_speedup(faster) - _effective_speedup(slower),
        "input": (
            "Improve this fused8 Triton candidate using the faster reference candidate and benchmark feedback.\n\n"
            f"Task: {task_id}\n"
            f"Slow source: {slower['source_type']} speedup {_fmt(_effective_speedup(slower))}\n"
            f"Fast source: {faster['source_type']} speedup {_fmt(_effective_speedup(faster))}\n"
        ),
    }


def _best_rows_by_task_and_source(runs: dict[str, dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for label, data in runs.items():
        repeatability = _load_repeatability_index(data["run_dir"])
        for record in data["bundle"]["candidate_records"]:
            row = _candidate_row(label, data["run_dir"], record, repeatability)
            if not row:
                continue
            key = (row["task_id"], row["source_type"])
            if key not in best or row["speedup_vs_eager"] > best[key]["speedup_vs_eager"]:
                best[key] = row
    return best


def _all_candidate_rows_from_runs(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, data in runs.items():
        repeatability = _load_repeatability_index(data["run_dir"])
        for record in data["bundle"]["candidate_records"]:
            row = _candidate_row(label, data["run_dir"], record, repeatability)
            if row:
                rows.append(row)
    return rows


def _final_stable_winners_from_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    winners: dict[str, dict[str, Any]] = {}
    for task_id in FUSED8_TASKS:
        candidates = [
            row
            for row in rows
            if row.get("task_id") == task_id and _stability_label(row) == "STABLE_WIN_ABOVE_EAGER"
        ]
        if candidates:
            winners[task_id] = max(candidates, key=_effective_speedup)
    return winners


def _stability_label(row: dict[str, Any]) -> str:
    repeat = row.get("repeatability") or {}
    stats = repeat.get("stats") or {}
    repeat_median = stats.get("median")
    single = row.get("speedup_vs_eager")
    if repeat_median is not None:
        if repeat.get("stable") and float(repeat_median) >= 1.0:
            return "STABLE_WIN_ABOVE_EAGER"
        if single is not None and float(single) >= 1.0:
            return "UNSTABLE"
    if single is not None and float(single) >= 1.0:
        return "SINGLE_RUN_ONLY"
    return "BELOW_EAGER"


def _effective_speedup(row: dict[str, Any]) -> float:
    repeat = row.get("repeatability") or {}
    stats = repeat.get("stats") or {}
    if stats.get("median") is not None and repeat.get("stable"):
        return float(stats["median"])
    return float(row.get("speedup_vs_eager") or 0.0)


def _load_repeatability_index(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = run_dir / "repeatability_results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in data.get("results", []):
        task = str(row.get("task_id") or "")
        candidate_path = str(row.get("candidate_path") or "")
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_path:
            index[(task, _path_key(candidate_path))] = row
        if candidate_id:
            index[(task, candidate_id)] = row
    return index


def _match_repeatability(record: dict[str, Any], index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    task = str(record.get("task_id") or "")
    path = str(record.get("candidate_path") or "")
    candidate_id = str(record.get("candidate_id") or "")
    return index.get((task, _path_key(path))) or index.get((task, candidate_id)) or {}


def _path_key(path: str) -> str:
    parts = Path(path).parts
    if "candidates" in parts:
        idx = parts.index("candidates")
        return "/".join(parts[idx:])
    return str(Path(path))


def _repeatability_status(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "repeatability_results.json"
    if not path.exists():
        return {"present": False, "rows": 0}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"present": True, "rows": len(data.get("results", [])), "top_k": data.get("top_k"), "repeats": data.get("repeats")}


def _run_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    records = [
        record
        for record in bundle["candidate_records"]
        if (record.get("benchmark_summary") or {}).get("speedup_vs_eager") is not None
    ]
    speedups = [float((record.get("benchmark_summary") or {})["speedup_vs_eager"]) for record in records]
    compile_speedups = [
        float((record.get("benchmark_summary") or {}).get("speedup_vs_torch_compile"))
        for record in records
        if (record.get("benchmark_summary") or {}).get("speedup_vs_torch_compile") is not None
    ]
    by_task = _best_records_by_task(records)
    return {
        "candidates": len(records),
        "verified": sum(1 for record in records if record.get("verification_passed")),
        "median_speedup": median(speedups) if speedups else None,
        "median_compile_speedup": median(compile_speedups) if compile_speedups else None,
        "tasks_gt_eager": sum(1 for record in by_task.values() if float((record.get("benchmark_summary") or {}).get("speedup_vs_eager", 0.0)) >= 1.0),
        "tasks_gt_compile": sum(1 for record in by_task.values() if float((record.get("benchmark_summary") or {}).get("speedup_vs_torch_compile", 0.0) or 0.0) >= 1.0),
    }


def _best_records_by_task(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        task = str(record.get("task_id") or "")
        speed = float((record.get("benchmark_summary") or {}).get("speedup_vs_eager") or 0.0)
        if task not in best or speed > float((best[task].get("benchmark_summary") or {}).get("speedup_vs_eager") or 0.0):
            best[task] = record
    return best


def _gemini_competitive_tasks(best_rows: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    tasks: list[str] = []
    for task_id in FUSED8_TASKS:
        template = best_rows.get((task_id, "template"))
        geminis = [best_rows.get((task_id, "gemini")), best_rows.get((task_id, "gemini_template_guided"))]
        geminis = [row for row in geminis if row]
        if not template or not geminis:
            continue
        best_gemini = max(geminis, key=_effective_speedup)
        if _effective_speedup(best_gemini) >= 0.9 * _effective_speedup(template):
            tasks.append(task_id)
    return tasks


def _template_dominated_tasks(best_rows: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    tasks: list[str] = []
    for task_id in FUSED8_TASKS:
        template = best_rows.get((task_id, "template"))
        geminis = [best_rows.get((task_id, "gemini")), best_rows.get((task_id, "gemini_template_guided"))]
        geminis = [row for row in geminis if row]
        if not template or not geminis:
            continue
        best_gemini = max(geminis, key=_effective_speedup)
        if _effective_speedup(template) > 1.1 * _effective_speedup(best_gemini):
            tasks.append(task_id)
    return tasks


def _template_metadata(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "template_family",
        "template_id",
        "block_size",
        "reduction_block_size",
        "num_warps",
        "num_stages",
        "contiguous_policy",
        "output_allocation_policy",
        "shape_specialized",
        "feature_dim_mode",
        "n_elements_mode",
    )
    return {key: record.get(key) for key in keys if record.get(key) is not None}


def _task_split(rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for filename, file_rows in rows.items():
        for row in file_rows:
            split[str(row.get("task_id"))][filename] += 1
    return {task: dict(counts) for task, counts in sorted(split.items())}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _readme(manifest: dict[str, Any]) -> str:
    lines = [
        "# OpenKernelForge Curated Fused8 Dataset",
        "",
        "Internal fused8 benchmark data only. This is not KernelBench and not a SOTA claim.",
        "",
        "## Counts",
        "",
    ]
    for filename, count in manifest["counts_by_file"].items():
        lines.append(f"- {filename}: {count}")
    lines.extend(["", "Stable-fast rows require repeat median speedup >= 1.0x and stable repeatability."])
    return "\n".join(lines) + "\n"


def _load_curated_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    return {filename: _read_jsonl_checked(root / filename, []) if (root / filename).exists() else [] for filename in CURATED_FILES}


def _load_manifest(root: Path, errors: list[str] | None = None) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        if errors is not None:
            errors.append("missing manifest.json")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if errors is not None:
            errors.append(f"manifest.json invalid JSON: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl_checked(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_no} invalid JSON: {exc}")
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                errors.append(f"{path.name}:{line_no} row is not an object")
    return rows


def _validate_curated_row(filename: str, line_no: int, row: dict[str, Any], errors: list[str]) -> None:
    required = ["task_id", "task_family", "source_type", "label"]
    for field in required:
        if field not in row:
            errors.append(f"{filename}:{line_no} missing {field}")
    if row.get("task_family") != "fused8":
        errors.append(f"{filename}:{line_no} task_family is not fused8")
    if filename in {
        "correct_fast_repeat_stable.jsonl",
        "correct_fast_single_run.jsonl",
        "correct_promising.jsonl",
        "rejected_or_unstable.jsonl",
    }:
        for field in ("candidate_code", "candidate_path", "benchmark"):
            if not row.get(field):
                errors.append(f"{filename}:{line_no} missing {field}")
    if filename.startswith("optimization_pairs_"):
        for field in ("slow_code", "fast_code", "slow_source_type", "fast_source_type"):
            if not row.get(field):
                errors.append(f"{filename}:{line_no} missing {field}")


def _curated_red_flags(rows_by_file: dict[str, list[dict[str, Any]]]) -> list[str]:
    flags: list[str] = []
    all_rows = [row for rows in rows_by_file.values() for row in rows]
    missing_code = sum(1 for row in all_rows if not row.get("candidate_code") and not row.get("fast_code"))
    missing_repeat = sum(
        1
        for row in rows_by_file.get("correct_fast_repeat_stable.jsonl", [])
        if not row.get("repeatability")
    )
    unstable_single = len(rows_by_file.get("correct_fast_single_run.jsonl", []))
    below_eager = len(rows_by_file.get("rejected_or_unstable.jsonl", []))
    stable_hashes = Counter(
        _code_hash(row.get("candidate_code") or "")
        for row in rows_by_file.get("correct_fast_repeat_stable.jsonl", [])
        if row.get("candidate_code")
    )
    duplicate_count = sum(1 for count in stable_hashes.values() if count > 1)
    if missing_code:
        flags.append(f"missing candidate code rows: {missing_code}")
    if missing_repeat:
        flags.append(f"stable-fast rows missing repeatability: {missing_repeat}")
    if unstable_single:
        flags.append(f"single-run-only candidates separated from stable-fast: {unstable_single}")
    if duplicate_count:
        flags.append(f"duplicate stable-fast code hashes: {duplicate_count}")
    if below_eager:
        flags.append(f"unstable/below-threshold candidates isolated: {below_eager}")
    return flags


def _source_summary(code: str) -> str:
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    flags = []
    if "@triton.jit" in code:
        flags.append("triton.jit")
    if "tl.sum" in code:
        flags.append("reduction")
    if "tl.maximum" in code:
        flags.append("relu-like")
    if "tl.exp" in code or "tl.sigmoid" in code:
        flags.append("exp/sigmoid")
    if "torch.empty_like" in code:
        flags.append("empty_like")
    elif "torch.empty" in code:
        flags.append("empty")
    return f"{len(lines)} non-empty lines; " + (", ".join(flags) if flags else "no static flags")


def _code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
