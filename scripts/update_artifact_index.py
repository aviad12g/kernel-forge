from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


INDEX_ITEMS = [
    ("rigorous deterministic fused8 template", "runs/20260520_155839", True),
    ("rigorous Gemini fused8 baseline", "runs/20260520_163344", True),
    ("rigorous OpenAI mini fused8 baseline", "runs/20260520_163607", True),
    ("rigorous fused8 model comparison", "runs/rigorous_fused8_model_comparison.md", True),
    ("deterministic fused8 template wide", "runs/20260519_213349_template_fused8_wide", True),
    ("Gemini fused8 baseline", "runs/20260519_215314_gemini_fused8_baseline", True),
    ("Gemini fused8 template-guided", "runs/20260519_215439_gemini_fused8_template_guided", True),
    ("OpenAI mini cheap", "runs/20260520_083300_openai_mini_fused8_cheap", True),
    ("GPT-5.5 cheap", "runs/20260520_085334_openai_gpt55_fused8_cheap", True),
    ("Qwen 7B local", "runs/20260520_114551_qwen7b_fused8_cheap", True),
    ("curated fused8 dataset", "datasets/fused8_curated_v1", True),
    ("final fused8 conclusion", "reports/fused8_phase11_conclusion.md", True),
    ("repeatability comparison", "reports/fused8_repeatability_comparison.md", True),
    ("Gemini/template comparison", "reports/fused8_gemini_vs_template_comparison.md", False),
    ("all-model comparison", "reports/fused8_all_model_comparison.md", False),
    ("KernelBench safe baseline validation", "runs/20260520_181052", True),
    ("KernelBench Gemini candidate pilot", "runs/20260520_202314", True),
    ("KernelBench candidate failure analysis", "runs/20260520_202314/kernelbench_candidate_failure_analysis.md", True),
    ("KernelBench failure taxonomy JSON", "runs/20260520_202314/kernelbench_failure_taxonomy.json", True),
    ("KernelBench repair subset", "runs/20260520_202314/kernelbench_repair_subset.md", True),
    ("KernelBench Gemini repair pass", "runs/20260520_213128", True),
    ("KernelBench repair comparison", "runs/kernelbench_gemini_repair1_comparison.md", True),
    ("KernelBench memory-safe selection config", "configs/kernelbench_l1_20task_rigorous_safe.yaml", True),
    ("KernelBench Gemini pilot config", "configs/kernelbench_l1_20task_gemini_rigorous.yaml", True),
    ("KernelBench repair config", "configs/kernelbench_l1_20task_gemini_repair1.yaml", True),
    ("KernelBench interpretation notes", "reports/kernelbench_interpretation_notes.md", True),
    ("KernelBench loss-win static analysis", "reports/kernelbench_loss_win_static_analysis.md", True),
    ("KernelBench profiler diagnostic status", "reports/profiling/kernelbench_loss_profiler_summary.md", False),
    ("Fused8 artifact recovery notes", "reports/fused8_artifact_recovery_notes.md", True),
    ("KernelBench adapter audit", "reports/kernelbench_adapter_audit.md", True),
    ("KernelBench current-policy re-audit", "reports/tables/kernelbench_historical_policy_reaudit.csv", True),
]

KERNELBENCH_HISTORICAL_LABELS = {
    "KernelBench safe baseline validation",
    "KernelBench Gemini candidate pilot",
    "KernelBench candidate failure analysis",
    "KernelBench failure taxonomy JSON",
    "KernelBench repair subset",
    "KernelBench Gemini repair pass",
    "KernelBench repair comparison",
    "KernelBench interpretation notes",
    "KernelBench loss-win static analysis",
    "KernelBench profiler diagnostic status",
}


def update_artifact_index(root: str | Path = ".", artifacts_dir: str | Path = "artifacts") -> Path:
    """Update reports/artifact_index.md using imported artifact status."""

    root = Path(root)
    artifacts = root / artifacts_dir
    imported = artifacts / "runpod_imports" if (artifacts / "runpod_imports").exists() else artifacts
    lines = [
        "# OpenKernelForge Artifact Index",
        "",
        "This index is generated from the current workspace. Missing artifacts are not inferred or fabricated.",
        "",
        f"- Imported artifact root: `{imported}`",
        "- KernelBench repo path used on RunPod: `/workspace/KernelBench`",
        "- KernelBench commit: `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`",
        "",
        "| Artifact | Location | Availability | Evidence status | Required |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, rel, required in INDEX_ITEMS:
        imported_path = imported / rel
        display_path, status = _location_and_status(
            imported_path,
            required,
            root,
            imported,
        )
        evidence = _evidence_status(label)
        lines.append(
            f"| {label} | `{display_path}` | {status} | {evidence} | "
            f"{'yes' if required else 'optional'} |"
        )
    lines.extend(
        [
            "",
            "## Generated Reports",
            "",
            f"- Technical report: `{root / 'reports/openkernelforge_technical_report.md'}`",
            f"- Reproducibility guide: `{root / 'reports/reproducibility.md'}`",
            f"- Paper PDF: `{root / 'paper/openkernelforge_paper.pdf'}`",
            f"- Artifact preservation plan: `{root / 'reports/artifact_preservation_plan.md'}`",
            f"- Historical KernelBench adapter audit: `{root / 'reports/kernelbench_adapter_audit.md'}`",
            f"- Historical candidate policy re-audit: `{root / 'reports/tables/kernelbench_historical_policy_reaudit.csv'}`",
            "",
        ]
    )
    out = root / "reports" / "artifact_index.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update OpenKernelForge artifact index from imported artifacts.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args(argv)
    path = update_artifact_index(args.root, args.artifacts_dir)
    print(f"Updated artifact index: {path}")
    return 0


def _location_and_status(
    path: Path,
    required: bool,
    root: Path,
    imported: Path,
) -> tuple[Path, str]:
    if path.exists():
        return path, "present under imported artifacts"
    rel = path.relative_to(imported) if path.is_relative_to(imported) else None
    workspace_path = root / rel if rel is not None else None
    if workspace_path is not None and workspace_path.exists():
        return workspace_path, "present in workspace"
    if required and (root / "reports/openkernelforge_technical_report.md").exists():
        return path, "summarized only"
    return path, "missing" if required else "optional missing"


def _evidence_status(label: str) -> str:
    if label in KERNELBENCH_HISTORICAL_LABELS:
        return "historical evaluator artifact; provisional"
    if label == "KernelBench repair config":
        return "historical parent-run provenance only"
    if label in {"KernelBench adapter audit", "KernelBench current-policy re-audit"}:
        return "current static audit"
    if label in {"KernelBench memory-safe selection config", "KernelBench Gemini pilot config"}:
        return "current code path; corrected CUDA result pending"
    return "supported or provenance artifact"


if __name__ == "__main__":
    raise SystemExit(main())
