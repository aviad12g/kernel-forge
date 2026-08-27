from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from openkernelforge.harness.policy import CandidatePolicyResult


DEFAULT_RUN = Path("runs/20260520_202314")
CURRENT_POLICY_VERSION = CandidatePolicyResult(passed=False).policy_version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze failed KernelBench Gemini candidates.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--max-repair", type=int, default=8)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    data = json.loads((run_dir / "kernelbench_l1_check.json").read_text(encoding="utf-8"))
    analysis = build_failure_analysis(run_dir, data, max_repair=args.max_repair)

    taxonomy_path = run_dir / "kernelbench_failure_taxonomy.json"
    taxonomy_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    report_path = run_dir / "kernelbench_candidate_failure_analysis.md"
    report_path.write_text(render_failure_report(analysis), encoding="utf-8")

    subset_path = run_dir / "kernelbench_repair_subset.md"
    subset_path.write_text(render_repair_subset(analysis), encoding="utf-8")

    print(f"Wrote {report_path}")
    print(f"Wrote {taxonomy_path}")
    print(f"Wrote {subset_path}")
    return 0


def build_failure_analysis(run_dir: Path, data: dict[str, Any], *, max_repair: int = 8) -> dict[str, Any]:
    failed = [
        analyze_failed_candidate(run_dir, item)
        for item in data.get("candidate_records", [])
        if not item.get("verification_passed")
    ]
    selected = select_repair_subset(failed, max_repair=max_repair)
    return {
        "run_dir": str(run_dir),
        "source_run_dir": data.get("run_dir"),
        "total_candidates": len(data.get("candidate_records", [])),
        "failed_candidates": len(failed),
        "summary": {
            "count_by_failure_category": dict(Counter(item["failure_category"] for item in failed)),
            "count_by_op_family": dict(Counter(item["op_family"] for item in failed)),
            "count_by_repairability": dict(Counter(item["repairability"] for item in failed)),
        },
        "likely_prompt_weaknesses": likely_prompt_weaknesses(failed),
        "failures": failed,
        "selected_for_repair": selected,
    }


def analyze_failed_candidate(run_dir: Path, item: dict[str, Any]) -> dict[str, Any]:
    verification = item.get("verification") or {}
    cases = verification.get("cases") or []
    error_text = "\n".join(
        str(part or "")
        for part in [
            verification.get("error"),
            item.get("failure_reason"),
            " ".join(str(case.get("message") or "") for case in cases),
            " ".join(str(case.get("error_type") or "") for case in cases),
        ]
    )
    log_path = item.get("error_log_path")
    if log_path and Path(log_path).exists():
        error_text += "\n" + Path(log_path).read_text(encoding="utf-8", errors="replace")

    category = classify_failure(item, error_text)
    repairability = classify_repairability(item, category, error_text)
    first_case = cases[0] if cases else {}
    return {
        "task_id": item.get("task_id"),
        "task_name": item.get("task_name"),
        "op_family": item.get("op_family"),
        "parent_run_dir": str(run_dir),
        "kernelbench_source_path": item.get("kernelbench_source_path"),
        "candidate_path": item.get("candidate_path"),
        "parent_policy_version": item.get("policy_version"),
        "parent_candidate_contract": item.get("candidate_contract"),
        "parent_reference_has_model_state": item.get("reference_has_model_state"),
        "policy_passed": item.get("policy_passed"),
        "policy_rejection_reason": item.get("policy_rejection_reason"),
        "verification_error": verification.get("error") or first_case.get("message") or item.get("failure_reason"),
        "exception_class": exception_class(error_text),
        "traceback_summary": short_text(error_text, 1600),
        "input_shapes": [case.get("shape") for case in cases if case.get("shape") is not None],
        "output_shape_expected": first_case.get("reference_shape"),
        "output_shape_actual": first_case.get("output_shape"),
        "max_abs_error": first_case.get("max_abs_error"),
        "max_rel_error": first_case.get("max_rel_error"),
        "failure_category": category,
        "repairability": repairability,
        "suggested_repair_instruction": suggested_repair_instruction(item, category, repairability),
        "parent_evidence_status": (
            "corrected_contract"
            if item.get("policy_version") == CURRENT_POLICY_VERSION
            and item.get("candidate_contract")
            else "historical_or_unversioned"
        ),
    }


def classify_failure(item: dict[str, Any], error_text: str) -> str:
    text = error_text.lower()
    verification = item.get("verification") or {}
    cases = verification.get("cases") or []
    if not item.get("policy_passed"):
        return "illegal torch fallback"
    extraction = item.get("extraction") or {}
    if extraction.get("error"):
        return "syntax/import error"
    if "outofmemoryerror" in text or "cuda out of memory" in text:
        return "timeout/OOM"
    if "modulenotfounderror" in text or "importerror" in text:
        return "syntax/import error"
    if "missing" in text and "required positional" in text:
        return "wrong number of inputs"
    if "takes" in text and "positional argument" in text:
        return "wrong function signature"
    if "compilationerror" in text or "triton.compiler" in text or "triton.language.semantic" in text:
        return "Triton compile error"
    if "reference_output = task.reference_fn" in error_text and "candidate_output" not in error_text:
        return "verification harness issue"
    for case in cases:
        if case.get("error_type") == "shape_mismatch":
            return "shape mismatch"
        if case.get("output_shape") and case.get("reference_shape") and case.get("output_shape") != case.get("reference_shape"):
            return "shape mismatch"
    if "values_not_close" in text or "torch.allclose failed" in text:
        return "numerical mismatch"
    if "attributeerror" in text or "runtimeerror" in text or "exception" in text:
        return "runtime exception"
    return "other"


def classify_repairability(item: dict[str, Any], category: str, error_text: str) -> str:
    family = str(item.get("op_family") or "")
    task_id = str(item.get("task_id") or "")
    if category in {"wrong function signature", "wrong number of inputs", "shape mismatch", "Triton compile error"}:
        return "high"
    if category == "numerical mismatch" and family in {"matmul", "loss"}:
        return "high"
    if category == "numerical mismatch" and family == "convolution":
        return "low"
    if category == "runtime exception" and "forward." in error_text:
        return "low"
    if category in {"timeout/OOM", "verification harness issue", "unsupported op pattern"}:
        return "low"
    if "conv" in task_id.lower() and family == "convolution":
        return "low"
    if category == "runtime exception":
        return "medium"
    return "medium"


def suggested_repair_instruction(item: dict[str, Any], category: str, repairability: str) -> str:
    family = str(item.get("op_family") or "")
    if category == "numerical mismatch" and family == "matmul":
        return "Use fp32 accumulation, exact indexing/strides, and implement the triangular/symmetric/transposed contract from the task source."
    if category == "numerical mismatch" and family == "loss":
        return "Match PyTorch reduction semantics exactly, including log/probability convention, reduction mode, and numerical stabilization."
    if category == "Triton compile error":
        return "Fix Triton pointer arithmetic and tl.load/tl.store masks; keep block sizes constexpr and avoid invalid pointer types."
    if category == "shape mismatch":
        return "Derive output shape from the task source and preserve the exact expected layout."
    if category == "runtime exception":
        return "Remove invalid state assumptions such as forward.weights; forward only receives get_inputs() arguments."
    if category == "timeout/OOM":
        return "Do not repair in this pass; first reduce memory pressure in the verification harness or pick a smaller task."
    if category == "verification harness issue":
        return "Do not repair the candidate until the reference/candidate comparison ambiguity is resolved."
    if family == "convolution":
        return "Avoid in this repair pass unless weights and convolution parameters are available to forward without calling the reference."
    return "Repair correctness against the KernelBench source and keep the implementation simple."


def select_repair_subset(failures: list[dict[str, Any]], *, max_repair: int) -> list[dict[str, Any]]:
    def score(item: dict[str, Any]) -> tuple[int, str]:
        category = item["failure_category"]
        family = item["op_family"]
        repairability = item["repairability"]
        value = 0
        if repairability == "high":
            value += 100
        elif repairability == "medium":
            value += 50
        if category == "numerical mismatch":
            value += 25
        if category == "Triton compile error":
            value += 20
        if family in {"matmul", "loss"}:
            value += 15
        if family == "pooling":
            value += 5
        if family == "convolution":
            value -= 75
        if category in {"timeout/OOM", "verification harness issue"}:
            value -= 100
        return (-value, str(item["task_id"]))

    candidates = [
        item
        for item in failures
        if item["repairability"] in {"high", "medium"}
        and item["failure_category"] not in {"timeout/OOM", "verification harness issue"}
        and item["op_family"] != "convolution"
    ]
    selected = sorted(candidates, key=score)[:max_repair]
    return selected


def likely_prompt_weaknesses(failures: list[dict[str, Any]]) -> list[str]:
    weaknesses: list[str] = []
    categories = Counter(item["failure_category"] for item in failures)
    families = Counter(item["op_family"] for item in failures)
    if categories.get("numerical mismatch", 0):
        weaknesses.append("Prompts need stronger emphasis on exact PyTorch semantics, reduction modes, and fp32 accumulation.")
    if families.get("convolution", 0):
        weaknesses.append("Convolution tasks expose hidden Model parameters; prompts must clarify which values are available to forward.")
    if categories.get("Triton compile error", 0):
        weaknesses.append("Prompts should call out valid Triton pointer arithmetic and constexpr block-size constraints.")
    if categories.get("runtime exception", 0):
        weaknesses.append("Prompts should explicitly forbid assuming attributes such as forward.weights or model state.")
    if categories.get("timeout/OOM", 0):
        weaknesses.append("Large-output tasks can fail during comparison; memory-risk tasks should be excluded from repair.")
    return weaknesses


def exception_class(text: str) -> str | None:
    matches = re.findall(r"([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))", text)
    return matches[-1] if matches else None


def short_text(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def render_failure_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# KernelBench Candidate Failure Analysis",
        "",
        f"- Run dir: `{analysis['run_dir']}`",
        f"- Total candidates: {analysis['total_candidates']}",
        f"- Failed candidates: {analysis['failed_candidates']}",
        "",
        "## Summary",
        "",
        "### Count By Failure Category",
        "",
    ]
    for key, count in sorted(analysis["summary"]["count_by_failure_category"].items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "### Count By Op Family", ""])
    for key, count in sorted(analysis["summary"]["count_by_op_family"].items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "### Count By Repairability", ""])
    for key, count in sorted(analysis["summary"]["count_by_repairability"].items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Likely Prompt Weaknesses", ""])
    for weakness in analysis["likely_prompt_weaknesses"]:
        lines.append(f"- {weakness}")
    lines.extend(
        [
            "",
            "## Failed Candidates",
            "",
            "| Task | Family | Category | Repairability | Exception | Expected | Actual | Suggested repair |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in analysis["failures"]:
        lines.append(
            "| `{task}` | {family} | `{category}` | {repairability} | {exception} | `{expected}` | `{actual}` | {instruction} |".format(
                task=item["task_id"],
                family=item["op_family"],
                category=item["failure_category"],
                repairability=item["repairability"],
                exception=item.get("exception_class") or "",
                expected=item.get("output_shape_expected"),
                actual=item.get("output_shape_actual"),
                instruction=short_text(item["suggested_repair_instruction"], 120),
            )
        )
    return "\n".join(lines) + "\n"


def render_repair_subset(analysis: dict[str, Any]) -> str:
    selected = analysis["selected_for_repair"]
    lines = [
        "# KernelBench Gemini Repair Subset",
        "",
        f"Source run: `{analysis['run_dir']}`",
        "",
        "Selected up to 8 failed tasks for one capped Gemini repair attempt. The selection prioritizes high-repairability numerical mismatches and clear Triton/runtime issues, while avoiding large convolution tasks and memory-risk cases.",
        "",
        "| Task | Family | Failure category | Why selected | Repair target | Prompt contents |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in selected:
        lines.append(
            "| `{task}` | {family} | `{category}` | {why} | {target} | Original task source, failed candidate, verification error, expected signature/output contract. |".format(
                task=item["task_id"],
                family=item["op_family"],
                category=item["failure_category"],
                why=("high repairability" if item["repairability"] == "high" else "medium repairability"),
                target=short_text(item["suggested_repair_instruction"], 140),
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
