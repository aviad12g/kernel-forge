"""Reports for template variants skipped before generation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def write_skipped_variants_artifacts(
    run_dir: str | Path,
    skipped_by_task: dict[str, list[dict[str, Any]]],
) -> tuple[Path, Path]:
    """Write skipped template variant JSONL and Markdown report."""

    run_path = Path(run_dir)
    rows = [
        {**row, "task_id": task_id}
        for task_id, rows_for_task in sorted(skipped_by_task.items())
        for row in rows_for_task
    ]
    jsonl_path = run_path / "skipped_variants.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    report_path = run_path / "skipped_variants_report.md"
    report_path.write_text(format_skipped_variants_report(rows), encoding="utf-8")
    return jsonl_path, report_path


def load_skipped_variants(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "skipped_variants.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def format_skipped_variants_report(rows: list[dict[str, Any]]) -> str:
    reason_counts = Counter(str(row.get("rejection_reason") or "unknown") for row in rows)
    task_counts = Counter(str(row.get("task_id") or "unknown") for row in rows)
    lines = [
        "# Skipped Template Variants",
        "",
        f"- Skipped invalid variants: {len(rows)}",
        "",
        "## Reasons",
        "",
    ]
    if reason_counts:
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## By Task", ""])
    if task_counts:
        for task_id, count in sorted(task_counts.items()):
            lines.append(f"- {task_id}: {count}")
    else:
        lines.append("- none")
    if rows:
        lines.extend(
            [
                "",
                "## Examples",
                "",
                "| Task | Reason | Metadata |",
                "| --- | --- | --- |",
            ]
        )
        for row in rows[:20]:
            metadata = row.get("template_metadata") or {}
            lines.append(
                f"| {row.get('task_id')} | {row.get('rejection_reason')} | "
                f"`{json.dumps(metadata, sort_keys=True)}` |"
            )
    lines.append("")
    return "\n".join(lines)
