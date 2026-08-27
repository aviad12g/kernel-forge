from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SECRET_PATTERNS = (
    re.compile(r"OPENAI_API_KEY\s*=\s*[^<\s][^\s]*"),
    re.compile(r"GEMINI_API_KEY\s*=\s*[^<\s][^\s]*"),
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"AIza[A-Za-z0-9_\-]{12,}"),
    re.compile(r"rpa_[A-Za-z0-9_\-]{20,}"),
)
REQUIRED_REPORTS = [
    "reports/openkernelforge_technical_report.md",
    "reports/reproducibility.md",
    "reports/artifact_index.md",
    "reports/artifact_preservation_plan.md",
    "reports/release_checklist.md",
]
REQUIRED_RUN_PREFIXES = [
    "20260520_155839",
    "20260520_163344",
    "20260520_163607",
    "20260519_213349",
    "20260519_215314",
    "20260519_215439",
    "20260520_083300",
    "20260520_085334",
    "20260520_114551",
    "20260520_202314",
    "20260520_213128",
]


def validate_research_package(
    root: str | Path = ".",
    artifacts_dir: str | Path = "artifacts",
    *,
    strict: bool = False,
) -> tuple[bool, list[str], list[str], Path]:
    """Validate reports and imported research artifacts without requiring GPU/API access."""

    root = Path(root)
    artifacts = root / artifacts_dir
    errors: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED_REPORTS:
        if not (root / rel).exists():
            errors.append(f"missing report: {rel}")

    imported = artifacts / "runpod_imports" if (artifacts / "runpod_imports").exists() else artifacts
    if artifacts.exists():
        _validate_imported_artifacts(imported, errors, warnings, strict=strict)
        _scan_for_secrets(artifacts, errors)
    else:
        warnings.append(f"{artifacts_dir} is not present; validating reports only")

    _scan_for_secrets(root / "reports", errors)
    report = _write_validation_report(root, artifacts, errors, warnings)
    return not errors, errors, warnings, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate OpenKernelForge research artifact package.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when required raw runs or the curated dataset are missing.",
    )
    args = parser.parse_args(argv)
    ok, errors, warnings, report = validate_research_package(
        args.root,
        args.artifacts_dir,
        strict=args.strict,
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print("Research package validation failed:")
        for error in errors:
            print(f"- {error}")
        print(f"Report: {report}")
        return 1
    print("Research package validation passed.")
    print(f"Report: {report}")
    return 0


def _validate_imported_artifacts(
    artifacts: Path,
    errors: list[str],
    warnings: list[str],
    *,
    strict: bool,
) -> None:
    for prefix in REQUIRED_RUN_PREFIXES:
        matches = list((artifacts / "runs").glob(f"{prefix}*"))
        if not matches:
            message = f"missing imported run artifact matching {prefix}"
            (errors if strict else warnings).append(message)
            continue
        for run_dir in matches:
            _validate_run_dir(run_dir, errors, warnings)

    dataset = artifacts / "datasets" / "fused8_curated_v1"
    if dataset.exists():
        if not (dataset / "manifest.json").exists():
            errors.append("imported curated dataset missing manifest.json")
        if not (dataset / "correct_fast_repeat_stable.jsonl").exists():
            errors.append("imported curated dataset missing correct_fast_repeat_stable.jsonl")
    else:
        (errors if strict else warnings).append("imported curated dataset missing")

    _verify_checksums(artifacts, errors, warnings, strict=strict)

    for path in (artifacts / "reports").glob("*.md") if (artifacts / "reports").exists() else []:
        if path.stat().st_size == 0:
            errors.append(f"empty imported report: {path}")


def _validate_run_dir(run_dir: Path, errors: list[str], warnings: list[str]) -> None:
    if not (run_dir / "results.jsonl").exists():
        errors.append(f"{run_dir} missing results.jsonl")
    else:
        _parse_jsonl(run_dir / "results.jsonl", errors)
    if not (run_dir / "environment_probe.json").exists():
        errors.append(f"{run_dir} missing environment_probe.json")

    is_kernelbench = (run_dir / "kernelbench_l1_check.json").exists()
    if is_kernelbench:
        if not (run_dir / "kernelbench_l1_check.md").exists():
            warnings.append(f"{run_dir} missing kernelbench_l1_check.md")
    elif not (run_dir / "fused8_report.md").exists() and not (run_dir / "analysis.md").exists():
        warnings.append(f"{run_dir} missing fused8_report.md and analysis.md")


def _parse_jsonl(path: Path, errors: list[str]) -> None:
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSONL {path}:{line_no}: {exc}")


def _verify_checksums(
    artifacts: Path,
    errors: list[str],
    warnings: list[str],
    *,
    strict: bool,
) -> None:
    checksum_path = artifacts / "SHA256SUMS"
    if not checksum_path.exists():
        (errors if strict else warnings).append("imported artifacts missing SHA256SUMS")
        return
    import hashlib

    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"invalid checksum row: {line!r}")
            continue
        path = (artifacts / relative).resolve()
        try:
            path.relative_to(artifacts.resolve())
        except ValueError:
            errors.append(f"unsafe checksum path: {relative}")
            continue
        if not path.is_file():
            errors.append(f"checksummed artifact missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"artifact checksum mismatch: {relative}")


def _scan_for_secrets(root: Path, errors: list[str]) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible secret pattern `{pattern.pattern}` found in {path}")
                break


def _write_validation_report(root: Path, artifacts: Path, errors: list[str], warnings: list[str]) -> Path:
    report_dir = artifacts if artifacts.exists() else root / "artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "validation_report.md"
    lines = [
        "# OpenKernelForge Research Package Validation",
        "",
        f"- Artifacts directory: `{artifacts}`",
        f"- Status: {'PASS' if not errors else 'FAIL'}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {error}" for error in errors) if errors else lines.append("- none")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
