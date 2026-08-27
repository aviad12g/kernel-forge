from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
OUT_CSV = TABLES / "label_threshold_sensitivity.csv"
OUT_NOTES = ROOT / "reports" / "label_threshold_sensitivity_notes.md"

TAUS = [0.95, 0.97, 0.98, 0.99]


HEADLINE_TASKS = [
    ("fused8", "bias_relu"),
    ("fused8", "residual"),
    ("fused8", "bias_gelu"),
    ("fused8", "rmsnorm"),
    ("KernelBench", "CrossEntropyLoss"),
    ("KernelBench", "TripletMarginLoss"),
    ("KernelBench repair", "KLDivLoss"),
]


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for study, task in HEADLINE_TASKS:
        artifact = _find_artifact(study, task)
        for tau in TAUS:
            rows.append(_row(study, task, tau, artifact))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "study",
                "task",
                "tau",
                "session_speedups_available",
                "artifact_label",
                "artifact_speedup_vs_eager",
                "label_at_tau",
                "artifact_source",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    OUT_NOTES.write_text(_notes(rows), encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_NOTES}")
    return 0


def _find_artifact(study: str, task: str) -> dict[str, str]:
    if study == "fused8":
        winner_rows = _read_csv(TABLES / "fused8_stable_winners.csv")
        template_rows = _read_csv(TABLES / "fused8_template_results.csv")
        row = next((r for r in winner_rows if r.get("task") == task), {})
        template = next((r for r in template_rows if r.get("task") == task), {})
        label = "BELOW_EAGER"
        if task == "bias_relu":
            label = "SINGLE_RUN_ONLY_WIN"
        elif row.get("stable_winner") not in {"", "none", "n/a", None}:
            label = "REPEAT_STABLE_WIN"
        return {
            "label": label,
            "speedup": row.get("repeat_median_speedup_vs_eager") or template.get("repeat_median_speedup_vs_eager", ""),
            "source": row.get("run_dir") or template.get("run_dir", ""),
            "session_speedups": "",
            "notes": "full per-session fused8 speedups are not preserved locally",
        }

    run = "20260520_202314" if study == "KernelBench" else "20260520_213128"
    p = ROOT / "artifacts" / "runpod_imports" / "runs" / run / "results.jsonl"
    if not p.exists():
        p = ROOT / "runs" / run / "results.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if task in record.get("task_id", ""):
                summary = record.get("benchmark_summary") or {}
                return {
                    "label": record.get("candidate_label", ""),
                    "speedup": str(summary.get("speedup_vs_eager", "")),
                    "source": str(p),
                    "session_speedups": ",".join(map(str, summary.get("session_speedups", []))) if isinstance(summary.get("session_speedups"), list) else "",
                    "notes": "KernelBench artifact preserves stable_above_eager at tau=0.98 but not per-session speedup vector",
                }
    return {
        "label": "artifact missing",
        "speedup": "",
        "source": "",
        "session_speedups": "",
        "notes": "artifact missing",
    }


def _row(study: str, task: str, tau: float, artifact: dict[str, str]) -> dict[str, str]:
    speedups = [float(x) for x in artifact.get("session_speedups", "").split(",") if x]
    if speedups:
        median = sorted(speedups)[len(speedups) // 2]
        label = "REPEAT_STABLE_WIN" if median >= 1.0 and all(s >= tau for s in speedups) else ("SINGLE_RUN_ONLY_WIN" if median >= 1.0 else "BELOW_EAGER")
        available = "yes"
        notes = "computed from preserved per-session speedups"
    else:
        label = "not preserved"
        available = "no"
        notes = artifact.get("notes", "per-session speedups not preserved")
        if tau == 0.98 and artifact.get("label"):
            notes += "; artifact label is preserved for implemented default tau/CV path"
    return {
        "study": study,
        "task": task,
        "tau": f"{tau:.2f}",
        "session_speedups_available": available,
        "artifact_label": artifact.get("label", ""),
        "artifact_speedup_vs_eager": artifact.get("speedup", ""),
        "label_at_tau": label,
        "artifact_source": artifact.get("source", ""),
        "notes": notes,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _notes(rows: list[dict[str, str]]) -> str:
    unavailable = [r for r in rows if r["session_speedups_available"] == "no"]
    return "\n".join(
        [
            "# Label Threshold Sensitivity Notes",
            "",
            "This analysis uses existing artifacts only. It does not rerun benchmarks.",
            "",
            f"Rows written: {len(rows)}.",
            f"Rows without preserved per-session speedups: {len(unavailable)}.",
            "",
            "Result: the local artifact package does not preserve the per-session speedup vectors needed to recompute headline labels at tau values 0.95, 0.97, 0.98, and 0.99. The artifacts preserve current labels and, for KernelBench, the stable_above_eager boolean computed by the implemented tau=0.98 path. No threshold-robustness claim is made.",
            "",
            "Future validation should preserve session_speedups for every candidate and rerun this script to compute threshold sensitivity directly.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
