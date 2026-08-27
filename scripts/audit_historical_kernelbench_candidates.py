from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

from openkernelforge.harness.policy import CandidatePolicyResult, check_candidate_policy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "artifacts" / "runpod_imports" / "runs"
RUN_IDS = ("20260520_202314", "20260520_213128")
CURRENT_POLICY_VERSION = CandidatePolicyResult(passed=False).policy_version


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Statically re-audit preserved KernelBench candidates under the current policy."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=ROOT / "reports" / "tables" / "kernelbench_historical_policy_reaudit.csv",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=ROOT / "reports" / "kernelbench_adapter_audit.md",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for run_id in RUN_IDS:
        run_dir = args.run_root / run_id
        if not run_dir.exists():
            missing.append(run_id)
            continue
        records = _candidate_records(run_dir / "results.jsonl")
        for path in sorted((run_dir / "candidates").glob("*/candidate_*.py")):
            task_id = path.parent.name
            record = records.get((task_id, path.stem), records.get((task_id, "candidate_000"), {}))
            source = path.read_text(encoding="utf-8")
            policy = check_candidate_policy(
                source,
                allow_torch_fallback=False,
                require_triton=True,
            )
            rows.append(
                {
                    "run_id": run_id,
                    "task_id": task_id,
                    "op_family": str(record.get("op_family") or "not preserved"),
                    "entrypoint": _entrypoint(source),
                    "historical_policy_pass": str(record.get("policy_passed", "not preserved")).lower(),
                    "current_policy_pass": str(policy.passed).lower(),
                    "current_rejection_reason": policy.rejection_reason or "",
                    "historical_verification_pass": str(
                        record.get("verification_passed", "not preserved")
                    ).lower(),
                    "candidate_path": str(path.relative_to(ROOT)),
                    "evidence_status": "historical_adapter_output_only",
                }
            )

    _write_csv(args.csv_out, rows)
    _write_report(args.report_out, rows, missing)
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.report_out}")
    return 0


def _candidate_records(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    records: dict[tuple[str, str], dict[str, object]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("record_type") != "kernelbench_candidate":
            continue
        task_id = str(record.get("task_id") or "")
        index = int(record.get("candidate_index") or 0)
        records[(task_id, f"candidate_{index:03d}")] = record
    return records


def _entrypoint(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "syntax_error"
    has_forward = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"
        for node in tree.body
    )
    has_model_new = any(isinstance(node, ast.ClassDef) and node.name == "ModelNew" for node in tree.body)
    if has_model_new:
        return "ModelNew"
    if has_forward:
        return "free_forward"
    return "missing"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "run_id",
        "task_id",
        "op_family",
        "entrypoint",
        "historical_policy_pass",
        "current_policy_pass",
        "current_rejection_reason",
        "historical_verification_pass",
        "candidate_path",
        "evidence_status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, str]], missing: list[str]) -> None:
    by_run = Counter(row["run_id"] for row in rows)
    passed_by_run = Counter(row["run_id"] for row in rows if row["current_policy_pass"] == "true")
    reasons = Counter(row["current_rejection_reason"] for row in rows if row["current_rejection_reason"])
    free_functions = sum(row["entrypoint"] == "free_forward" for row in rows)

    lines = [
        "# Historical KernelBench Adapter Audit",
        "",
        "## Evidence status",
        "",
        "The preserved KernelBench runs are retained for evaluator auditing, not as supported correctness or performance evidence. The historical adapter required a free `forward(*args)` candidate receiving only `get_inputs()`, although every official task that defines `Model` requires a `ModelNew` instance initialized through `get_init_inputs()`. Parameterized tasks additionally had no route to the reference state. The adapter also reconstructed and transferred the reference `Model` inside each reference call. These defects invalidate model-accuracy and speedup interpretations of the affected rows.",
        "",
        "The corrected adapter materializes one seeded initialization snapshot and uses persistent reference `Model` and candidate `ModelNew` instances outside verification and timing loops. Every official `Model` task rejects free-function candidates. The completed corrected workshop campaign is reported separately; the historical rows audited here remain excluded.",
        "",
        "## Static policy re-audit",
        "",
        f"This re-audit parses preserved source with the current `{CURRENT_POLICY_VERSION}` policy. It does not import or execute candidates and does not repair the historical task-contract or timing defects.",
        "",
    ]
    for run_id in RUN_IDS:
        if run_id in missing:
            lines.append(f"- `{run_id}`: artifacts missing")
        else:
            lines.append(
                f"- `{run_id}`: {passed_by_run[run_id]}/{by_run[run_id]} preserved candidates pass the current strict policy."
            )
    lines.extend(
        [
            f"- Free-function entry points: {free_functions}/{len(rows)} preserved candidates.",
            "",
            "Current-policy rejection reasons:",
            "",
        ]
    )
    if reasons:
        for reason, count in sorted(reasons.items()):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none recorded")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        "The historical `policy pass` counts cannot be compared directly with current policy counts because the policy implementation changed. Even a source that passes the current AST guardrail is not thereby valid for an official KernelBench `Model` task without the `ModelNew` contract, and AST checks are not a security sandbox. The machine-readable rows are in `reports/tables/kernelbench_historical_policy_reaudit.csv`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
