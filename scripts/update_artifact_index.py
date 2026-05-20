from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


INDEX_ITEMS = [
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
]


def update_artifact_index(root: str | Path = ".", artifacts_dir: str | Path = "artifacts") -> Path:
    """Update reports/artifact_index.md using imported artifact status."""

    root = Path(root)
    artifacts = root / artifacts_dir
    lines = [
        "# OpenKernelForge Artifact Index",
        "",
        "This index is generated from the current workspace. Missing artifacts are not inferred or fabricated.",
        "",
        f"- Imported artifact root: `{artifacts}`",
        "",
        "| Artifact | Imported path | Status | Required |",
        "| --- | --- | --- | --- |",
    ]
    for label, rel, required in INDEX_ITEMS:
        path = artifacts / rel
        status = _status(path, required, root)
        lines.append(f"| {label} | `{path}` | {status} | {'yes' if required else 'optional'} |")
    lines.extend(
        [
            "",
            "## Generated Reports",
            "",
            f"- Technical report: `{root / 'reports/openkernelforge_technical_report.md'}`",
            f"- Reproducibility guide: `{root / 'reports/reproducibility.md'}`",
            f"- Artifact preservation plan: `{root / 'reports/artifact_preservation_plan.md'}`",
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


def _status(path: Path, required: bool, root: Path) -> str:
    if path.exists():
        return "present locally"
    if required and (root / "reports/openkernelforge_technical_report.md").exists():
        return "summarized only"
    return "missing" if required else "optional missing"


if __name__ == "__main__":
    raise SystemExit(main())
