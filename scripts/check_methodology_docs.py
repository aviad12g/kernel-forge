from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from openkernelforge.harness.policy import CandidatePolicyResult

CURRENT_POLICY_VERSION = CandidatePolicyResult(passed=False).policy_version

REQUIRED_FILES = [
    "docs/methodology/repeatability_label_spec.md",
    "docs/methodology/static_policy_checks.md",
    "docs/methodology/timing_protocol.md",
    "docs/methodology/environment.md",
    "docs/methodology/prompt_templates.md",
    "docs/methodology/fused8_tasks.md",
    "docs/methodology/kernelbench_repairability.md",
    "docs/methodology/kernelbench_adapter_contract.md",
    "paper/overleaf/sections/methodology.tex",
    "paper/overleaf/sections/prompt_appendix.tex",
    "paper/overleaf/tables/static_policy_checks_table.tex",
    "paper/overleaf/tables/fused8_shapes_tolerances_table.tex",
]

REPEATABILITY_LABELS = [
    "REPEAT_STABLE_WIN",
    "SINGLE_RUN_ONLY_WIN",
    "BELOW_EAGER",
    "UNSTABLE",
    "INSUFFICIENT_DATA",
]

POLICY_REASONS = [
    "missing_forward",
    "imports_reference_or_task_module",
    "disallowed_from_torch_import",
    "disallowed_import_alias",
    "import_time_call",
    "calls_suspicious_function",
    "obvious_torch_fallback",
]

UNSUPPORTED_POSITIVE_CLAIMS = [
    "achieves state of the art",
    "achieves sota",
    "full kernelbench result",
    "full kernelbench benchmark",
]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def check_methodology_docs(root: Path = ROOT) -> tuple[bool, list[str]]:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        if not (root / rel).exists():
            errors.append(f"missing required methodology file: {rel}")

    if errors:
        return False, errors

    repeat_doc = (root / "docs/methodology/repeatability_label_spec.md").read_text(encoding="utf-8")
    method_tex = (root / "paper/overleaf/sections/methodology.tex").read_text(encoding="utf-8")
    for label in REPEATABILITY_LABELS:
        if label not in repeat_doc:
            errors.append(f"repeatability spec missing label {label}")
        escaped = label.replace("_", r"\_")
        if label not in method_tex and escaped not in method_tex:
            errors.append(f"methodology.tex missing label {label}")

    policy_doc = (root / "docs/methodology/static_policy_checks.md").read_text(encoding="utf-8")
    policy_table = (root / "paper/overleaf/tables/static_policy_checks_table.tex").read_text(encoding="utf-8")
    for reason in POLICY_REASONS:
        if reason not in policy_doc:
            errors.append(f"static policy docs missing implemented reason {reason}")
        escaped = reason.replace("_", r"\_")
        if reason not in policy_table and escaped not in policy_table:
            errors.append(f"static policy table missing implemented reason {reason}")
    if CURRENT_POLICY_VERSION not in policy_doc:
        errors.append(
            "static policy docs must identify policy version " + CURRENT_POLICY_VERSION
        )

    timing_doc = (root / "docs/methodology/timing_protocol.md").read_text(encoding="utf-8").lower()
    for phrase in ["cudaeventtimer", "120 per session", "100 measured samples", "128 mb", "cache-state perturbation"]:
        if phrase not in timing_doc:
            errors.append(f"timing protocol missing phrase: {phrase}")

    environment_doc = (root / "docs/methodology/environment.md").read_text(encoding="utf-8").lower()
    for phrase in ["rtx 5090", "torch 2.8.0+cu128", "triton 3.4.0", "locked gpu clocks"]:
        if phrase not in environment_doc:
            errors.append(f"environment doc missing phrase: {phrase}")

    prompt_doc = (root / "docs/methodology/prompt_templates.md").read_text(encoding="utf-8")
    for phrase in ["gemini-3.1-flash-lite", "gpt-5.4-mini", "candidate_provider = gemini_repair"]:
        if phrase not in prompt_doc:
            errors.append(f"prompt template docs missing phrase: {phrase}")

    fused8_doc = (root / "docs/methodology/fused8_tasks.md").read_text(encoding="utf-8")
    for phrase in ["bias_relu", "rmsnorm_small", "[4096,1024]", "2e-4"]:
        if phrase not in fused8_doc:
            errors.append(f"fused8 task docs missing phrase: {phrase}")

    repair_doc = (root / "docs/methodology/kernelbench_repairability.md").read_text(encoding="utf-8")
    for phrase in ["high repairability", "max_repair = 8", "KLDivLoss"]:
        if phrase not in repair_doc:
            errors.append(f"repairability docs missing phrase: {phrase}")

    adapter_doc = (root / "docs/methodology/kernelbench_adapter_contract.md").read_text(encoding="utf-8")
    for phrase in [
        "Every official KernelBench task that defines `Model`",
        "get_init_inputs()` once under fixed seed `0`",
        "persistent",
        "input side effects",
        "PyTorch's `meta` device",
        "provisional",
        "not an operating-system sandbox",
    ]:
        if phrase not in adapter_doc:
            errors.append(f"KernelBench adapter contract docs missing phrase: {phrase}")

    prompt_appendix = (root / "paper/overleaf/sections/prompt_appendix.tex").read_text(encoding="utf-8")
    if "every official task that defines \\code{Model}" not in prompt_appendix:
        errors.append("prompt appendix does not require ModelNew for every official Model task")
    if "Every official KernelBench task that defines \\code{Model}" not in method_tex:
        errors.append("methodology.tex does not state the uniform official ModelNew contract")
    forbidden_contract_phrases = [
        "Stateless tasks may retain a module-level",
        "Tasks with parameters or buffers reject a free function",
        "stateful tasks reject free functions",
    ]
    combined_contract_docs = "\n".join([adapter_doc, prompt_appendix, method_tex])
    for phrase in forbidden_contract_phrases:
        if phrase.lower() in combined_contract_docs.lower():
            errors.append(f"obsolete partial ModelNew contract remains in methodology: {phrase}")

    overleaf_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / "paper/overleaf").rglob("*.tex"))
    ).lower()
    if "todo citation" in overleaf_text:
        errors.append("Overleaf source contains TODO citation text")
    for phrase in UNSUPPORTED_POSITIVE_CLAIMS:
        negated = (
            f"not a {phrase}" in overleaf_text
            or f"no {phrase}" in overleaf_text
            or f"does not claim {phrase}" in overleaf_text
            or f"not designed to estimate {phrase.replace('result', 'performance')}" in overleaf_text
        )
        if phrase in overleaf_text and not negated:
            errors.append(f"possible unsupported claim in Overleaf source: {phrase}")

    return not errors, errors


def main(argv: list[str] | None = None) -> int:
    root = ROOT
    args = list(argv or sys.argv[1:])
    if args:
        if len(args) == 2 and args[0] == "--root":
            root = Path(args[1]).resolve()
        else:
            print("usage: python scripts/check_methodology_docs.py [--root <repo>]")
            return 2

    ok, errors = check_methodology_docs(root)
    if not ok:
        print("Methodology documentation check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Methodology documentation check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
