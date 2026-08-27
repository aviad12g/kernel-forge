from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = [
    "README.md",
    "paper/paper.md",
    "paper/methodology.md",
    "paper/experiments.md",
    "paper/limitations.md",
    "reports/openkernelforge_technical_report.md",
    "reports/reproducibility.md",
    "paper/overleaf/sections/abstract.tex",
    "paper/overleaf/sections/introduction.tex",
    "paper/overleaf/sections/methodology.tex",
    "paper/overleaf/sections/experiments.tex",
    "paper/overleaf/sections/discussion.tex",
    "paper/overleaf/sections/limitations.tex",
    "paper/overleaf/sections/conclusion.tex",
    "paper/workshop2026/sections/abstract.tex",
]

FORBIDDEN_STALE_PHRASES = [
    "In the capped KernelBench L1 pilot, Gemini finds",
    "The combined pilot has 4 unique correct tasks",
    "showing that the external adapter and repair paths work",
    "The three preserved KernelBench loss candidates keep their repeat-stable labels",
    "all three stable KernelBench wins",
    "sandboxed import",
]

REQUIRED_TEXT = {
    "README.md": [
        "historical KernelBench pilot rows",
        "not merged with the corrected results",
        "None of the 10 frozen task winners exceeded eager",
    ],
    "paper/overleaf/sections/abstract.tex": [
        "post-hoc adapter audit",
        "exclude their correctness rates and speedups",
    ],
    "paper/overleaf/sections/experiments.tex": [
        "not estimates of Gemini accuracy or KernelBench performance",
        "Corrected GPU revalidation is required",
    ],
    "paper/overleaf/sections/limitations.tex": [
        "no KernelBench correctness or speedup claim",
    ],
    "reports/openkernelforge_technical_report.md": [
        "historical KernelBench pilot remains an evaluator-audit artifact",
        "No winner crossed the 2% eager margin",
    ],
    "paper/workshop2026/sections/abstract.tex": [
        "27 of 141",
        "none of the frozen task winners exceeded eager",
        "three of four tasks appeared above the margin",
        "only two confirmed",
    ],
}

REQUIRED_ARTIFACTS = [
    "docs/methodology/kernelbench_adapter_contract.md",
    "reports/kernelbench_adapter_audit.md",
    "reports/tables/kernelbench_historical_policy_reaudit.csv",
    "artifacts/workshop2026/holdout_campaign/analysis/aggregate_promotion_summary.json",
    "artifacts/workshop2026/multiplicity/campaign/selection_multiplicity.csv",
    "artifacts/workshop2026/near_threshold_multiplicity_v3/campaign/selection_multiplicity.csv",
    "reports/workshop2026_corrected_results.md",
]


def check_claim_status(root: Path = ROOT) -> tuple[bool, list[str]]:
    errors: list[str] = []
    texts: dict[str, str] = {}
    for rel in PUBLIC_FILES:
        path = root / rel
        if not path.exists():
            errors.append(f"missing public claim file: {rel}")
            continue
        texts[rel] = path.read_text(encoding="utf-8")

    for rel, text in texts.items():
        normalized = " ".join(text.split())
        for phrase in FORBIDDEN_STALE_PHRASES:
            if phrase.lower() in normalized.lower():
                errors.append(f"stale KernelBench claim phrase in {rel}: {phrase}")

    for rel, phrases in REQUIRED_TEXT.items():
        text = " ".join(texts.get(rel, "").split())
        for phrase in phrases:
            if phrase.lower() not in text.lower():
                errors.append(f"{rel} missing evidence-status phrase: {phrase}")

    for rel in REQUIRED_ARTIFACTS:
        if not (root / rel).exists():
            errors.append(f"missing KernelBench audit artifact: {rel}")

    summary_table = root / "paper/overleaf/tables/kernelbench_summary_table.tex"
    if not summary_table.exists():
        errors.append("missing generated KernelBench summary table")
    elif "Post-hoc KernelBench adapter audit" not in summary_table.read_text(encoding="utf-8"):
        errors.append("KernelBench summary table does not present the adapter audit")

    return not errors, errors


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    root = ROOT
    if args:
        if len(args) == 2 and args[0] == "--root":
            root = Path(args[1]).resolve()
        else:
            print("usage: python scripts/check_kernelbench_claim_status.py [--root <repo>]")
            return 2
    ok, errors = check_claim_status(root)
    if not ok:
        print("KernelBench claim-status check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("KernelBench claim-status check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
