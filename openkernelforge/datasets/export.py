"""Export OpenKernelForge runs into inspectable JSONL datasets."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openkernelforge.datasets.schemas import BASE_REQUIRED_FIELDS, DATASET_FILES
from openkernelforge.agents.prompt_templates import build_task_prompt
from openkernelforge.reports.failure_taxonomy import classify_candidate_record
from openkernelforge.reports.run_data import load_run_bundle, read_artifact
from openkernelforge.reports.skipped_variants import load_skipped_variants
from openkernelforge.tasks.simple_tasks import get_task


def export_dataset(run_dir: str | Path, out_dir: str | Path) -> Path:
    """Export run artifacts into SFT, repair, optimization, and rejected JSONL files."""

    run_path = Path(run_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bundle = load_run_bundle(run_path)
    candidates = bundle["candidate_records"]
    classifications = {id(record): classify_candidate_record(record) for record in candidates}
    rows = {filename: [] for filename in DATASET_FILES}
    skipped_variant_rows = _skipped_variant_rows(run_path)

    for record in candidates:
        classification = classifications[id(record)]
        base = _base_row(run_path, record, classification)
        if record.get("policy_passed") and record.get("verification_passed"):
            row = dict(base)
            row["target"] = row["candidate_code"]
            row["target_type"] = "sft_raw"
            rows["sft_raw.jsonl"].append(row)
        elif classification.suggested_dataset_split == "rejected" or not record.get("verification_passed"):
            row = dict(base)
            row["target_type"] = "rejected"
            rows["rejected.jsonl"].append(row)

    for row in _repair_rows(run_path, candidates):
        rows["repair.jsonl"].append(row)
    for row in _optimization_rows(run_path, candidates):
        rows["optimization.jsonl"].append(row)

    counts: dict[str, int] = {}
    for filename, file_rows in rows.items():
        _write_jsonl(out_path / filename, file_rows)
        counts[filename] = len(file_rows)
    if skipped_variant_rows:
        _write_jsonl(out_path / "skipped_variants.jsonl", skipped_variant_rows)
        counts["skipped_variants.jsonl"] = len(skipped_variant_rows)

    failure_counts = Counter(
        classify_candidate_record(record).failure_type for record in candidates
    )
    manifest = {
        "source_run_dir": str(run_path),
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "counts_by_file": counts,
        "counts_by_failure_type": dict(sorted(failure_counts.items())),
        "run_kind": _run_kind(candidates, bundle["config"]),
        "warnings": _manifest_warnings(candidates, bundle["config"]),
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_path / "README.md").write_text(_dataset_readme(manifest), encoding="utf-8")
    return out_path


def validate_dataset(dataset_dir: str | Path) -> tuple[bool, list[str]]:
    """Validate exported dataset files and manifest counts."""

    root = Path(dataset_dir)
    errors: list[str] = []
    manifest_path = root / "manifest.json"
    readme_path = root / "README.md"
    if not manifest_path.exists():
        errors.append("missing manifest.json")
        manifest = {}
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"manifest.json is invalid JSON: {exc}")
            manifest = {}
    if not readme_path.exists():
        errors.append("missing README.md")

    observed_counts: dict[str, int] = {}
    for filename in DATASET_FILES:
        path = root / filename
        if not path.exists():
            errors.append(f"missing {filename}")
            observed_counts[filename] = 0
            continue
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{filename}:{line_no} invalid JSON: {exc}")
                    continue
                count += 1
                _validate_row(filename, line_no, row, errors)
        observed_counts[filename] = count

    manifest_counts = manifest.get("counts_by_file", {}) if isinstance(manifest, dict) else {}
    for filename, count in observed_counts.items():
        if manifest_counts.get(filename) != count:
            errors.append(
                f"manifest count mismatch for {filename}: "
                f"manifest={manifest_counts.get(filename)} observed={count}"
            )
    skipped_path = root / "skipped_variants.jsonl"
    if skipped_path.exists():
        skipped_count = 0
        with skipped_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"skipped_variants.jsonl:{line_no} invalid JSON: {exc}")
                    continue
                skipped_count += 1
                for field in ("task_id", "template_metadata", "rejection_reason", "source_type"):
                    if field not in row:
                        errors.append(f"skipped_variants.jsonl:{line_no} missing field {field}")
        if manifest_counts.get("skipped_variants.jsonl") != skipped_count:
            errors.append(
                "manifest count mismatch for skipped_variants.jsonl: "
                f"manifest={manifest_counts.get('skipped_variants.jsonl')} observed={skipped_count}"
            )
    return not errors, errors


def _base_row(
    run_path: Path,
    record: dict[str, Any],
    classification,
) -> dict[str, Any]:
    task_id = str(record.get("task_id") or "")
    try:
        task = get_task(task_id)
        description = task.description
        fallback_prompt = build_task_prompt(task)
        task_metadata = task.metadata
    except KeyError:
        description = ""
        fallback_prompt = ""
        task_metadata = {}
    prompt = read_artifact(record.get("prompt_path"), run_dir=run_path) or fallback_prompt
    return {
        "task_id": task_id,
        "task_name": record.get("task_name") or task_id,
        "task_description": description,
        "task_family": task_metadata.get("task_family"),
        "shape_metadata": task_metadata.get("shape_metadata"),
        "prompt": prompt,
        "raw_response": read_artifact(record.get("response_path"), run_dir=run_path),
        "candidate_code": read_artifact(record.get("candidate_path"), run_dir=run_path),
        "policy_result": _policy_result(record),
        "verification_result": record.get("verification_result") or record.get("verification_summary") or {},
        "benchmark_result": record.get("benchmark_result") or record.get("benchmark_summary"),
        "failure_type": classification.failure_type,
        "short_reason": classification.short_reason,
        "source_run_dir": str(run_path),
        "prompt_path": record.get("prompt_path"),
        "response_path": record.get("response_path"),
        "candidate_path": record.get("candidate_path"),
        "created_at": record.get("created_at"),
        "target_type": classification.suggested_dataset_split,
        "source_type": _source_type(record, prompt),
        "generation_stage": record.get("generation_stage", "initial"),
        "search_round": record.get("search_round"),
        "parent_candidate_path": record.get("parent_candidate_path"),
        "parent_speedup_vs_eager": record.get("parent_speedup_vs_eager"),
        "improved_over_parent": record.get("improved_over_parent"),
        "target_reached": record.get("target_reached"),
        "template_family": record.get("template_family"),
        "task_family_candidate": record.get("task_family"),
        "template_id": record.get("template_id"),
        "block_size": record.get("block_size"),
        "reduction_block_size": record.get("reduction_block_size"),
        "reduction_axis": record.get("reduction_axis"),
        "num_warps": record.get("num_warps"),
        "num_stages": record.get("num_stages"),
        "contiguous_policy": record.get("contiguous_policy"),
        "output_allocation_policy": record.get("output_allocation_policy"),
        "shape_specialized": record.get("shape_specialized"),
        "feature_dim_mode": record.get("feature_dim_mode"),
        "n_elements_mode": record.get("n_elements_mode"),
        "total_possible_variants": record.get("total_possible_variants"),
        "valid_possible_variants": record.get("valid_possible_variants"),
        "generated_valid_variants": record.get("generated_valid_variants"),
        "skipped_invalid_variants": record.get("skipped_invalid_variants"),
        "skipped_reasons": record.get("skipped_reasons"),
        "variant_validation": record.get("variant_validation"),
        "actually_generated_variants": record.get("actually_generated_variants"),
        "grid_sampling": record.get("grid_sampling"),
        "sort_order": record.get("sort_order"),
        "grid_was_capped": record.get("grid_was_capped"),
        "leaderboard_rank": record.get("leaderboard_rank"),
        "focused_seed_run": record.get("focused_seed_run"),
        "focused_seed_candidate_path": record.get("focused_seed_candidate_path"),
        "focused_rank": record.get("focused_rank"),
        "speedup_vs_compile": (record.get("benchmark_summary") or {}).get("speedup_vs_torch_compile"),
        "template_source_path": record.get("template_source_path"),
        "copied_from_template_id": record.get("copied_from_template_id"),
        "requested_block_size": record.get("requested_block_size"),
        "requested_num_warps": record.get("requested_num_warps"),
        "requested_contiguous_policy": record.get("requested_contiguous_policy"),
        "preserved_template_structure_score": record.get("preserved_template_structure_score"),
        "template_preservation": record.get("template_preservation"),
        "extra_torch_ops_detected": record.get("extra_torch_ops_detected"),
        "fallback_detected": record.get("fallback_detected"),
        "source_template_benchmark": record.get("source_template_benchmark"),
        "source_template_speedup_vs_eager": record.get("source_template_speedup_vs_eager"),
        "candidate_benchmark": record.get("benchmark_summary") or {},
        "delta_vs_source_template": record.get("delta_vs_source_template"),
    }


def _repair_rows(run_path: Path, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_task = _by_task_sorted(candidates)
    for task_id, records in by_task.items():
        correct = [record for record in records if record.get("policy_passed") and record.get("verification_passed")]
        if not correct:
            continue
        first_correct = correct[0]
        target_code = read_artifact(first_correct.get("candidate_path"), run_dir=run_path)
        for failed in records:
            if failed is first_correct:
                break
            if failed.get("verification_passed"):
                continue
            classification = classify_candidate_record(failed)
            base = _base_row(run_path, failed, classification)
            base.update(
                {
                    "target_type": "repair",
                    "input": _repair_input(base),
                    "broken_code": base["candidate_code"],
                    "target": target_code,
                    "target_candidate_path": first_correct.get("candidate_path"),
                    "target_candidate_id": first_correct.get("candidate_id"),
                }
            )
            rows.append(base)
    return rows


def _optimization_rows(run_path: Path, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str | None, str | None]] = set()
    by_candidate_id = {record.get("candidate_id"): record for record in candidates}
    by_candidate_path = {record.get("candidate_path"): record for record in candidates}
    for child in candidates:
        if child.get("generation_stage") != "template_copy":
            continue
        if not (child.get("policy_passed") and child.get("verification_passed")):
            continue
        preservation = child.get("template_preservation") or {}
        score = child.get("preserved_template_structure_score")
        if preservation and not preservation.get("passed"):
            continue
        if score is not None and float(score) < 70:
            continue
        source_template_path = child.get("template_source_path")
        pair_key = (source_template_path, child.get("candidate_path"))
        seen_pairs.add(pair_key)
        classification = classify_candidate_record(child)
        base = _base_row(run_path, child, classification)
        template_code = read_artifact(source_template_path, run_dir=run_path)
        child_code = base["candidate_code"]
        base.update(
            {
                "target_type": "template_copy_optimization",
                "input": _template_copy_optimization_input(base, template_code),
                "slow_code": template_code,
                "fast_code": child_code,
                "source_template_code": template_code,
                "copied_child_code": child_code,
                "target": child_code,
                "target_candidate_path": child.get("candidate_path"),
                "target_candidate_id": child.get("candidate_id"),
                "parent_benchmark": child.get("source_template_benchmark") or {},
                "child_benchmark": child.get("benchmark_summary") or {},
                "speedup_delta": child.get("delta_vs_source_template"),
                "optimization_prompt": base["prompt"],
                "generation_stage": "template_copy",
                "target_reached": child.get("target_reached"),
            }
        )
        rows.append(base)

    for child in candidates:
        if child.get("generation_stage") != "performance_search":
            continue
        if not (child.get("policy_passed") and child.get("verification_passed")):
            continue
        child_benchmark = child.get("benchmark_summary") or {}
        child_speedup = child_benchmark.get("speedup_vs_eager")
        parent_path = child.get("parent_candidate_path")
        parent = by_candidate_id.get(child.get("parent_candidate_id")) or by_candidate_path.get(parent_path)
        parent_benchmark = (parent or {}).get("benchmark_summary") or {}
        parent_speedup = child.get("parent_speedup_vs_eager") or parent_benchmark.get("speedup_vs_eager")
        if child_speedup is None or parent_speedup is None:
            continue
        speedup_delta = float(child_speedup) - float(parent_speedup)
        if speedup_delta <= 0:
            continue
        pair_key = (parent_path, child.get("candidate_path"))
        seen_pairs.add(pair_key)
        classification = classify_candidate_record(child)
        base = _base_row(run_path, child, classification)
        parent_code = read_artifact(parent_path, run_dir=run_path)
        child_code = base["candidate_code"]
        base.update(
            {
                "target_type": "optimization",
                "input": _performance_optimization_input(base, parent_code, parent_benchmark, child_benchmark),
                "slow_code": parent_code,
                "fast_code": child_code,
                "parent_slow_code": parent_code,
                "optimized_child_code": child_code,
                "target": child_code,
                "target_candidate_path": child.get("candidate_path"),
                "target_candidate_id": child.get("candidate_id"),
                "parent_candidate_id": child.get("parent_candidate_id"),
                "parent_benchmark": parent_benchmark,
                "child_benchmark": child_benchmark,
                "speedup_delta": speedup_delta,
                "optimization_prompt": base["prompt"],
                "generation_stage": "performance_search",
                "target_reached": child.get("target_reached"),
            }
        )
        rows.append(base)

    by_task = _by_task_sorted(candidates)
    for task_id, records in by_task.items():
        correct = [
            record
            for record in records
            if record.get("policy_passed")
            and record.get("verification_passed")
            and (record.get("benchmark_summary") or {}).get("candidate_median_ms") is not None
        ]
        if len(correct) < 2:
            continue
        sorted_correct = sorted(
            correct,
            key=lambda r: float((r.get("benchmark_summary") or {}).get("candidate_median_ms")),
        )
        fastest = sorted_correct[0]
        for slower in sorted_correct[1:]:
            pair_key = (slower.get("candidate_path"), fastest.get("candidate_path"))
            if pair_key in seen_pairs:
                continue
            fast_ms = float((fastest.get("benchmark_summary") or {}).get("candidate_median_ms"))
            slow_ms = float((slower.get("benchmark_summary") or {}).get("candidate_median_ms"))
            if slow_ms <= fast_ms * 1.05:
                continue
            classification = classify_candidate_record(slower)
            base = _base_row(run_path, slower, classification)
            fast_code = read_artifact(fastest.get("candidate_path"), run_dir=run_path)
            base.update(
                {
                    "target_type": "optimization",
                    "input": _optimization_input(base, slow_ms, fast_ms),
                    "slow_code": base["candidate_code"],
                    "fast_code": fast_code,
                    "target": fast_code,
                    "target_candidate_path": fastest.get("candidate_path"),
                    "target_candidate_id": fastest.get("candidate_id"),
                    "slow_candidate_median_ms": slow_ms,
                    "fast_candidate_median_ms": fast_ms,
                    "parent_benchmark": slower.get("benchmark_summary") or {},
                    "child_benchmark": fastest.get("benchmark_summary") or {},
                    "speedup_delta": _speedup_delta(slower, fastest),
                    "generation_stage": fastest.get("generation_stage", "initial"),
                    "target_reached": fastest.get("target_reached"),
                }
            )
            rows.append(base)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _skipped_variant_rows(run_path: Path) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "source_run_dir": str(run_path),
            "target_type": "skipped_variant",
            "source_type": "template_variant_validation",
        }
        for row in load_skipped_variants(run_path)
    ]


def _validate_row(filename: str, line_no: int, row: dict[str, Any], errors: list[str]) -> None:
    for field in BASE_REQUIRED_FIELDS:
        if field not in row:
            errors.append(f"{filename}:{line_no} missing field {field}")
    if not row.get("prompt"):
        errors.append(f"{filename}:{line_no} empty prompt")
    if filename in {"sft_raw.jsonl", "repair.jsonl", "optimization.jsonl"} and not row.get("target"):
        errors.append(f"{filename}:{line_no} empty target")
    if filename == "repair.jsonl":
        if not row.get("broken_code"):
            errors.append(f"{filename}:{line_no} empty broken_code")
        if not row.get("target"):
            errors.append(f"{filename}:{line_no} empty repair target")
    if filename == "optimization.jsonl":
        if not row.get("slow_code"):
            errors.append(f"{filename}:{line_no} empty slow_code")
        if not row.get("fast_code"):
            errors.append(f"{filename}:{line_no} empty fast_code")


def _policy_result(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("policy_result") or {
        "passed": record.get("policy_passed"),
        "warnings": record.get("policy_warnings") or [],
        "rejection_reason": record.get("policy_rejection_reason"),
    }


def _source_type(record: dict[str, Any], prompt: str) -> str:
    if record.get("generation_stage") == "template_copy":
        return "template_copy"
    if record.get("generation_stage") in {
        "template_baseline",
        "template_focused_sweep",
        "template_focused_clean",
    } or record.get("template_id"):
        return "template"
    if "Best deterministic template context" in prompt:
        return "template_guided"
    return "llm" if record.get("agent_type") == "llm" or record.get("backend") == "openai_compatible" else "template"


def _by_task_sorted(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        by_task[str(record.get("task_id"))].append(record)
    for records in by_task.values():
        records.sort(key=lambda r: (r.get("attempt_index") or 0, r.get("candidate_index") or 0))
    return by_task


def _repair_input(row: dict[str, Any]) -> str:
    return (
        "Repair this OpenKernelForge candidate.\n\n"
        f"Prompt:\n{row['prompt']}\n\n"
        f"Broken code:\n{row['candidate_code']}\n\n"
        f"Feedback:\n{row['short_reason']}\n{json.dumps(row['verification_result'])}\n"
    )


def _optimization_input(row: dict[str, Any], slow_ms: float, fast_ms: float) -> str:
    return (
        "Optimize this correct OpenKernelForge candidate.\n\n"
        f"Prompt:\n{row['prompt']}\n\n"
        f"Correct but slower code:\n{row['candidate_code']}\n\n"
        f"Benchmark feedback: candidate median {slow_ms:.6f} ms; faster target {fast_ms:.6f} ms.\n"
    )


def _performance_optimization_input(
    row: dict[str, Any],
    parent_code: str,
    parent_benchmark: dict[str, Any],
    child_benchmark: dict[str, Any],
) -> str:
    return (
        "Optimize this correct OpenKernelForge candidate.\n\n"
        f"Optimization prompt:\n{row['prompt']}\n\n"
        f"Parent correct-but-slow code:\n{parent_code}\n\n"
        f"Parent benchmark:\n{json.dumps(parent_benchmark)}\n\n"
        f"Optimized child benchmark:\n{json.dumps(child_benchmark)}\n"
    )


def _template_copy_optimization_input(row: dict[str, Any], template_code: str) -> str:
    requested = {
        "block_size": row.get("requested_block_size"),
        "num_warps": row.get("requested_num_warps"),
        "contiguous_policy": row.get("requested_contiguous_policy"),
    }
    return (
        "Copy/adapt this known-good Triton template with the requested parameter change.\n\n"
        f"Optimization prompt:\n{row['prompt']}\n\n"
        f"Source template code:\n{template_code}\n\n"
        f"Requested parameters:\n{json.dumps(requested)}\n\n"
        f"Source template benchmark:\n{json.dumps(row.get('source_template_benchmark') or {})}\n\n"
        f"Copied candidate benchmark:\n{json.dumps(row.get('candidate_benchmark') or {})}\n"
    )


def _speedup_delta(slower: dict[str, Any], faster: dict[str, Any]) -> float | None:
    slow = (slower.get("benchmark_summary") or {}).get("speedup_vs_eager")
    fast = (faster.get("benchmark_summary") or {}).get("speedup_vs_eager")
    if slow is None or fast is None:
        return None
    return float(fast) - float(slow)


def _run_kind(candidates: list[dict[str, Any]], config: dict[str, Any]) -> str:
    agent = (config.get("agent") or {}).get("type")
    backend = (config.get("agent") or {}).get("backend")
    if any(record.get("generation_stage") == "template_copy" for record in candidates):
        return "template_copy_model_candidate"
    if agent == "template" or any(str(record.get("generation_stage", "")).startswith("template_") for record in candidates):
        return "template_baseline"
    if agent == "dummy" or backend in {"fake", None}:
        return "harness_only"
    return "real_model_candidate"


def _manifest_warnings(candidates: list[dict[str, Any]], config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if _run_kind(candidates, config) == "harness_only":
        warnings.append("dummy/fake run: useful for pipeline tests, not training-quality model data")
    if _run_kind(candidates, config) == "template_baseline":
        warnings.append("template baseline: deterministic non-model data, useful as a performance reference")
    if not any(record.get("verification_passed") for record in candidates):
        warnings.append("no verified candidates exported to sft_raw")
    return warnings


def _dataset_readme(manifest: dict[str, Any]) -> str:
    return (
        "# OpenKernelForge Dataset Export\n\n"
        f"- Source run: `{manifest['source_run_dir']}`\n"
        f"- Exported at: {manifest['export_timestamp']}\n"
        f"- Run kind: {manifest['run_kind']}\n\n"
        "## Counts\n\n"
        + "\n".join(f"- {name}: {count}" for name, count in manifest["counts_by_file"].items())
        + "\n\nRows are JSONL and intended for inspection before any training use.\n"
    )
