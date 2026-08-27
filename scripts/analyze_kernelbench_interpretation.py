from __future__ import annotations

import ast
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
REPORT = ROOT / "reports" / "kernelbench_interpretation_notes.md"
LOSS_STATIC_REPORT = ROOT / "reports" / "kernelbench_loss_win_static_analysis.md"
COMPILE_TIME_REPORT = ROOT / "reports" / "compile_time_notes.md"
ONE_SHOT_RUN = ROOT / "runs" / "20260520_202314"
REPAIR_RUN = ROOT / "runs" / "20260520_213128"


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    one_shot_check = _load_json(ONE_SHOT_RUN / "kernelbench_l1_check.json")
    one_shot_records = _load_jsonl(ONE_SHOT_RUN / "results.jsonl")
    repair_records = _load_jsonl(REPAIR_RUN / "results.jsonl")
    taxonomy = _load_json(ONE_SHOT_RUN / "kernelbench_failure_taxonomy.json")

    family_rows = _family_outcomes(one_shot_records, repair_records)
    repairability_rows = _repairability_criteria(taxonomy)
    loss_rows = _loss_win_interpretation(one_shot_records, repair_records)
    eager_rows = _eager_baseline_notes()
    fused8_eager_rows = _fused8_eager_baseline_notes()
    compile_rows = _compile_time_summary(one_shot_records, repair_records)
    memory_rows = _memory_filter_rows(one_shot_check)

    _write_csv(TABLES / "kernelbench_family_outcomes_detailed.csv", family_rows)
    _write_csv(TABLES / "kernelbench_repairability_criteria.csv", repairability_rows)
    _write_csv(TABLES / "kernelbench_loss_win_interpretation.csv", loss_rows)
    _write_csv(TABLES / "kernelbench_eager_baseline_notes.csv", eager_rows)
    _write_csv(TABLES / "fused8_eager_baseline_notes.csv", fused8_eager_rows)
    _write_csv(TABLES / "compile_time_summary.csv", compile_rows)
    _write_csv(TABLES / "kernelbench_memory_filter_summary.csv", memory_rows)
    _write_notes(family_rows, repairability_rows, loss_rows, eager_rows, compile_rows, memory_rows)
    _write_loss_static_report(loss_rows)
    _write_compile_time_notes(compile_rows)

    for path in [
        TABLES / "kernelbench_family_outcomes_detailed.csv",
        TABLES / "kernelbench_repairability_criteria.csv",
        TABLES / "kernelbench_loss_win_interpretation.csv",
        TABLES / "kernelbench_eager_baseline_notes.csv",
        TABLES / "fused8_eager_baseline_notes.csv",
        TABLES / "compile_time_summary.csv",
        TABLES / "kernelbench_memory_filter_summary.csv",
        REPORT,
        LOSS_STATIC_REPORT,
        COMPILE_TIME_REPORT,
    ]:
        print(f"Wrote {path}")
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
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


def _family_outcomes(
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
    combined_correct: dict[str, set[str]] = {family: set() for family in families}
    combined_stable: dict[str, set[str]] = {family: set() for family in families}
    for record in one_shot_records + repair_records:
        family = str(record.get("op_family"))
        if family not in combined_correct:
            combined_correct[family] = set()
            combined_stable[family] = set()
        task = str(record.get("task_id"))
        if record.get("verification_passed"):
            combined_correct[family].add(task)
        if record.get("candidate_label") == "REPEAT_STABLE_WIN":
            combined_stable[family].add(task)

    interpretations = {
        "loss": "affected evaluator recorded its stable labels in this family",
        "matmul": "affected evaluator recorded one verification and no stable label",
        "convolution": "obsolete free-function contract also omitted reference state",
        "pooling": "affected evaluator recorded no verification",
    }
    rows = []
    for family in families:
        rows.append(
            {
                "family": family,
                "selected_tasks": selected[family],
                "one_shot_verified": one_verified[family],
                "one_shot_stable": one_stable[family],
                "repair_attempted": repair_attempted[family],
                "repair_verified": repair_verified[family],
                "combined_correct": len(combined_correct.get(family, set())),
                "combined_stable": len(combined_stable.get(family, set())),
                "interpretation": interpretations[family],
                "evidence_status": "historical_adapter_output_only",
            }
        )
    return rows


def _repairability_criteria(taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    summary = (taxonomy.get("summary") or {}).get("count_by_repairability") or {}
    return [
        {
            "repairability": "high",
            "operational_criterion": "wrong signature/input count, shape mismatch, Triton compile error, or numerical mismatch in matmul/loss tasks",
            "included_categories": "numerical mismatch in matmul/loss; localized Triton compile errors",
            "excluded_categories": "timeout/OOM, harness issue, convolution, broad semantic failures",
            "selected_count": 8,
            "observed_total_count": summary.get("high", 8),
            "notes": "historical selection provenance; not repair-effectiveness evidence",
            "evidence_status": "historical_adapter_output_only",
        },
        {
            "repairability": "medium",
            "operational_criterion": "runtime exception or other category not explicitly high/low",
            "included_categories": "not selected in the reported repair pass",
            "excluded_categories": "not applicable",
            "selected_count": 0,
            "observed_total_count": summary.get("medium", 0),
            "notes": "historical analyzer category; absent from the initial taxonomy summary",
            "evidence_status": "historical_adapter_output_only",
        },
        {
            "repairability": "low",
            "operational_criterion": "convolution numerical mismatch, timeout/OOM, verification harness issue, unsupported pattern, or invalid forward state",
            "included_categories": "not selected",
            "excluded_categories": "excluded from repair subset",
            "selected_count": 0,
            "observed_total_count": summary.get("low", 9),
            "notes": "historical selection provenance only",
            "evidence_status": "historical_adapter_output_only",
        },
    ]


def _loss_win_interpretation(
    one_shot_records: list[dict[str, Any]],
    repair_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = [
        record
        for record in one_shot_records + repair_records
        if record.get("op_family") == "loss" and record.get("candidate_label") == "REPEAT_STABLE_WIN"
    ]
    rows = []
    for record in records:
        task = _task_alias(str(record.get("task_name") or record.get("task_id")))
        candidate_path = ROOT / str(record.get("candidate_path", ""))
        prompt_path = ROOT / str(record.get("prompt_path", ""))
        source = candidate_path.read_text(encoding="utf-8") if candidate_path.exists() else ""
        mechanism = _loss_mechanism(task, source)
        caveat = "historical source pattern only; invalid reference lifecycle prevents performance attribution"
        if task == "KLDivLoss":
            caveat += "; candidate still uses torch.log and torch sum outside Triton"
        rows.append(
            {
                "task": task,
                "speedup_vs_eager": _fmt_speed((record.get("benchmark_summary") or {}).get("speedup_vs_eager")),
                "speedup_vs_compile": _fmt_speed((record.get("benchmark_summary") or {}).get("speedup_vs_torch_compile")),
                "likely_mechanism": mechanism,
                "evidence_source": _evidence_source(candidate_path, prompt_path),
                "confidence": "source present" if source else "low",
                "caveat": caveat,
                "evidence_status": "historical_adapter_output_only",
            }
        )
    return rows or [
        {
            "task": "not available",
            "speedup_vs_eager": "not available",
            "speedup_vs_compile": "not available",
            "likely_mechanism": "not available",
            "evidence_source": "verified loss candidate artifacts missing",
            "confidence": "low",
            "caveat": "not available",
            "evidence_status": "historical_adapter_output_only",
        }
    ]


def _torch_calls(source: str) -> list[str]:
    if not source:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _name_of(node.func)
            if name.startswith("torch.") or name.startswith("F.") or name.endswith(".mean") or name.endswith(".sum"):
                calls.append(name)
    return sorted(set(calls))


def _name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_of(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _loss_mechanism(task: str, source: str) -> str:
    if task == "CrossEntropyLoss":
        return "source contains row-wise log-sum-exp and target gather followed by a Torch mean"
    if task == "TripletMarginLoss":
        return "source contains one Triton kernel for distances and hinge loss followed by a Torch mean"
    if task == "KLDivLoss":
        return "source contains a Triton elementwise KL term plus Torch log and reduction"
    return "historical loss-candidate source pattern"


def _evidence_source(candidate_path: Path, prompt_path: Path) -> str:
    parts = []
    if candidate_path.exists():
        parts.append(str(candidate_path.relative_to(ROOT)))
    if prompt_path.exists():
        parts.append(str(prompt_path.relative_to(ROOT)))
    return "; ".join(parts) if parts else "not available"


def _eager_baseline_notes() -> list[dict[str, Any]]:
    return [
        {
            "task_or_family": "loss",
            "likely_eager_path": "torch.nn.functional loss functions and reductions",
            "evidence": "historical task source and profiler files",
            "confidence": "source-level only",
            "caveat": "affected reference lifecycle prevents performance attribution",
            "evidence_status": "historical_adapter_output_only",
        },
        {
            "task_or_family": "matmul",
            "likely_eager_path": "ATen matmul/elementwise paths, often backed by cuBLAS for dense matmul tasks",
            "evidence": "KernelBench task names and prompt sources are matmul variants",
            "confidence": "medium",
            "caveat": "diagonal/triangular variants may use elementwise or masking paths rather than plain GEMM",
            "evidence_status": "historical_adapter_output_only",
        },
        {
            "task_or_family": "convolution",
            "likely_eager_path": "torch.nn convolution modules, typically ATen/cuDNN CUDA kernels",
            "evidence": "KernelBench task sources instantiate Conv2d/Conv3d/transposed convolution modules",
            "confidence": "medium",
            "caveat": "exact backend algorithm not measured",
            "evidence_status": "historical_adapter_output_only",
        },
        {
            "task_or_family": "pooling",
            "likely_eager_path": "torch.nn pooling modules using ATen CUDA pooling kernels",
            "evidence": "KernelBench task sources instantiate MaxPool/AveragePool modules",
            "confidence": "medium",
            "caveat": "exact kernel path not measured",
            "evidence_status": "historical_adapter_output_only",
        },
    ]


def _fused8_eager_baseline_notes() -> list[dict[str, Any]]:
    return [
        {
            "task_or_family": "elementwise fused8",
            "likely_eager_path": "ATen elementwise kernels and broadcasting paths",
            "evidence": "fused8 task definitions use torch.relu, sigmoid, add, and broadcasting",
            "confidence": "medium",
            "caveat": "qualitative source inspection only; no profiler traces in local package",
        },
        {
            "task_or_family": "row_sum",
            "likely_eager_path": "ATen reduction kernel",
            "evidence": "reference uses sum over final dimension",
            "confidence": "medium",
            "caveat": "exact kernel path not recorded",
        },
        {
            "task_or_family": "layernorm/rmsnorm",
            "likely_eager_path": "ATen normalization and reduction operations",
            "evidence": "reference computes mean/variance or RMS over feature dimension",
            "confidence": "medium",
            "caveat": "no Nsight or torch.profiler attribution is preserved",
        },
    ]


def _compile_time_summary(
    one_shot_records: list[dict[str, Any]],
    repair_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for stage, records in [("one-shot Gemini", one_shot_records), ("repair1", repair_records)]:
        verified = [record for record in records if record.get("verification_passed")]
        compile_values = [
            (record.get("benchmark") or {}).get("compile_time_ms")
            for record in verified
            if (record.get("benchmark") or {}).get("compile_time_ms") is not None
        ]
        runtime_values = [
            (record.get("benchmark") or {}).get("runtime_only_ms")
            for record in verified
            if (record.get("benchmark") or {}).get("runtime_only_ms") is not None
        ]
        rows.append(
            {
                "stage": stage,
                "verified_candidates": len(verified),
                "compile_time_ms_available": len(compile_values),
                "compile_time_ms_summary": _median_summary(compile_values) if compile_values else "not preserved",
                "runtime_only_ms_available": len(runtime_values),
                "runtime_only_ms_summary": _median_summary(runtime_values) if runtime_values else "not preserved",
                "notes": "historical fields use obsolete accounting or are null; not interpreted",
                "evidence_status": "historical_adapter_output_only",
            }
        )
    return rows


def _median_summary(values: list[float]) -> str:
    values = sorted(float(v) for v in values)
    if not values:
        return "not preserved"
    mid = len(values) // 2
    if len(values) % 2:
        median = values[mid]
    else:
        median = (values[mid - 1] + values[mid]) / 2
    return f"median {median:.4f}"


def _memory_filter_rows(check: dict[str, Any]) -> list[dict[str, Any]]:
    records = check.get("records") or []
    skipped = [record for record in records if record.get("skipped")]
    selected = [record for record in records if not record.get("skipped")]
    skipped_by_family = Counter(str(record.get("op_family")) for record in skipped)
    selected_by_family = Counter(str(record.get("op_family")) for record in selected)
    families = sorted(set(skipped_by_family) | set(selected_by_family))
    rows = [
        {
            "category": "overall",
            "family": "all",
            "count": check.get("tasks_loaded", "not available"),
            "shape_examples": "not applicable",
            "interpretation": "historical adapter loaded official L1 pool",
            "evidence_status": "historical_adapter_output_only",
        },
        {
            "category": "selected_feasible",
            "family": "all",
            "count": len(selected),
            "shape_examples": "see selected task appendix",
            "interpretation": "historical subset after lower-bound filter",
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
                "interpretation": "historical selected family count",
                "evidence_status": "historical_adapter_output_only",
            }
        )
        rows.append(
            {
                "category": "skipped_memory_cap",
                "family": family,
                "count": skipped_by_family[family],
                "shape_examples": _shape_examples([r for r in skipped if str(r.get("op_family")) == family]),
                "interpretation": "historical filter count",
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
    return "; ".join(examples) if examples else "not available"


def _fmt_speed(value: Any) -> str:
    try:
        return f"{float(value):.3f}x"
    except (TypeError, ValueError):
        return "not available"


def _task_alias(task: str) -> str:
    if "CrossEntropyLoss" in task:
        return "CrossEntropyLoss"
    if "TripletMarginLoss" in task:
        return "TripletMarginLoss"
    if "KLDivLoss" in task:
        return "KLDivLoss"
    return task


def _write_notes(
    family_rows: list[dict[str, Any]],
    repairability_rows: list[dict[str, Any]],
    loss_rows: list[dict[str, Any]],
    eager_rows: list[dict[str, Any]],
    compile_rows: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# KernelBench Interpretation Notes",
        "",
        "This note summarizes artifacts from the affected historical KernelBench adapter. It does not execute candidates. Counts and timings are audit metadata, not model-accuracy or performance evidence.",
        "",
        "## Family-Level Outcomes",
        "",
        "| Family | Selected | One-shot verified | One-shot stable | Repair attempted | Repair verified | Combined correct | Interpretation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in family_rows:
        lines.append(
            f"| {row['family']} | {row['selected_tasks']} | {row['one_shot_verified']} | {row['one_shot_stable']} | "
            f"{row['repair_attempted']} | {row['repair_verified']} | {row['combined_correct']} | {row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Historical Loss-Candidate Source Patterns",
            "",
            "Source patterns are listed for auditability. The invalid reference lifecycle prevents mechanism attribution from the historical profiler or timing rows.",
            "",
            "| Task | Speedup vs eager | Likely mechanism | Caveat |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in loss_rows:
        lines.append(f"| {row['task']} | {row['speedup_vs_eager']} | {row['likely_mechanism']} | {row['caveat']} |")
    lines.extend(
        [
            "",
            "## Repairability",
            "",
            "High repairability is a historical selection heuristic, not evidence that repair is effective.",
            "",
            "## Eager and Compile Baselines",
            "",
            "Eager-path notes are qualitative. Historical compile fields use obsolete accounting or are null and are not interpreted.",
            "",
            "## Memory Filtering",
            "",
            "The historical filter was a lower-bound selection heuristic, not a complete peak-memory estimate.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_loss_static_report(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Historical KernelBench Loss-Candidate Source Audit",
        "",
        "This report inspects preserved source from the affected historical adapter. The old task-state and reference lifecycle invalidate performance and mechanism attribution; speed fields are retained only to identify their source records.",
        "",
        "| Task | Speedup vs eager | Speedup vs compile | Likely mechanism | Confidence | Caveat |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['speedup_vs_eager']} | {row['speedup_vs_compile']} | "
            f"{row['likely_mechanism']} | {row['confidence']} | {row['caveat']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: the source contains plausible fusion patterns, but the affected evaluator cannot establish that those patterns caused a valid speedup. Corrected candidate verification, timing, and profiling are required.",
        ]
    )
    LOSS_STATIC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_compile_time_notes(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Compile-Time Availability Notes",
        "",
        "Historical KernelBench compile fields used obsolete accounting or are null. They are retained for schema auditing and are not used in the paper.",
        "",
        "| Stage | Verified | Compile-time fields | Runtime-only fields | Notes |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['verified_candidates']} | {row['compile_time_ms_available']} | "
            f"{row['runtime_only_ms_available']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "The paper therefore does not analyze compile-cost amortization or deployment cost. This matters especially for `torch.compile max-autotune`, where compilation can be expensive relative to repeated runtime calls.",
        ]
    )
    COMPILE_TIME_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
