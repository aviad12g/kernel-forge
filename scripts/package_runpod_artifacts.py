from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUN_ROOT_FILES = {
    "config.yaml",
    "environment_probe.json",
    "results.jsonl",
    "summary.md",
    "analysis.md",
    "real_run_review.md",
    "fused8_report.md",
    "repeatability_report.md",
    "repeatability_results.json",
    "performance_search_report.md",
    "gpu_candidate_debrief.md",
    "profiler_lite_report.md",
    "template_autotune_report.md",
    "template_copy_report.md",
    "focused_sweep_report.md",
    "kernelbench_l1_check.md",
    "kernelbench_l1_check.json",
    "kernelbench_l1_pilot_report.md",
    "kernelbench_candidate_failure_analysis.md",
    "kernelbench_failure_taxonomy.json",
    "kernelbench_repair_subset.md",
}
RUN_DIRS = {"candidates", "prompts", "responses", "logs"}
DATASET_SUFFIXES = {".jsonl", ".json", ".md", ".txt"}
EXCLUDED_NAMES = {".git", "__pycache__", ".env"}
SECRET_PATTERNS = (
    re.compile(r"OPENAI_API_KEY\s*=\s*[^<\s][^\s]*"),
    re.compile(r"GEMINI_API_KEY\s*=\s*[^<\s][^\s]*"),
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"AIza[A-Za-z0-9_\-]{12,}"),
    re.compile(r"rpa_[A-Za-z0-9_\-]{20,}"),
)

ARTIFACT_SPECS: list[dict[str, Any]] = [
    {
        "source": "runs/20260520_155839",
        "dest": "runs/20260520_155839_template_fused8_rigorous",
        "required": True,
        "kind": "run",
        "why": "rigorous CUDA-event deterministic fused8 template benchmark",
    },
    {
        "source": "runs/20260520_163344",
        "dest": "runs/20260520_163344_gemini_fused8_rigorous",
        "required": True,
        "kind": "run",
        "why": "rigorous CUDA-event Gemini fused8 baseline",
    },
    {
        "source": "runs/20260520_163607",
        "dest": "runs/20260520_163607_openai_mini_fused8_rigorous",
        "required": True,
        "kind": "run",
        "why": "rigorous CUDA-event OpenAI mini fused8 baseline",
    },
    {
        "source": "runs/20260519_213349",
        "dest": "runs/20260519_213349_template_fused8_wide",
        "required": True,
        "kind": "run",
        "why": "deterministic fused8 template floor",
    },
    {
        "source": "runs/20260519_215314",
        "dest": "runs/20260519_215314_gemini_fused8_baseline",
        "required": True,
        "kind": "run",
        "why": "Gemini fused8 baseline",
    },
    {
        "source": "runs/20260519_215439",
        "dest": "runs/20260519_215439_gemini_fused8_template_guided",
        "required": True,
        "kind": "run",
        "why": "Gemini template-guided fused8 run",
    },
    {
        "source": "runs/20260520_083300",
        "dest": "runs/20260520_083300_openai_mini_fused8_cheap",
        "required": True,
        "kind": "run",
        "why": "OpenAI mini cheap fused8 baseline",
    },
    {
        "source": "runs/20260520_085334",
        "dest": "runs/20260520_085334_openai_gpt55_fused8_cheap",
        "required": True,
        "kind": "run",
        "why": "GPT-5.5 cheap fused8 baseline",
    },
    {
        "source": "runs/20260520_114551",
        "dest": "runs/20260520_114551_qwen7b_fused8_cheap",
        "required": True,
        "kind": "run",
        "why": "local Qwen 7B cheap fused8 baseline",
    },
    {
        "source": "runs/20260520_202314",
        "dest": "runs/20260520_202314_kernelbench_gemini_pilot",
        "required": True,
        "kind": "run",
        "why": "capped KernelBench L1 one-shot pilot and failure taxonomy",
    },
    {
        "source": "runs/20260520_213128",
        "dest": "runs/20260520_213128_kernelbench_repair1",
        "required": True,
        "kind": "run",
        "why": "capped KernelBench L1 repair iteration",
    },
    {
        "source": "datasets/fused8_curated_v1",
        "dest": "datasets/fused8_curated_v1",
        "required": True,
        "kind": "dataset",
        "why": "curated repeatability-aware fused8 dataset",
    },
    {
        "source": "runs/rigorous_fused8_model_comparison.md",
        "dest": "reports/rigorous_fused8_model_comparison.md",
        "required": True,
        "kind": "file",
        "why": "rigorous template/Gemini/OpenAI mini model comparison",
    },
    {
        "source": "runs/fused8_phase11_conclusion.md",
        "dest": "reports/fused8_phase11_conclusion.md",
        "required": True,
        "kind": "file",
        "why": "final fused8 conclusion",
    },
    {
        "source": "runs/fused8_repeatability_comparison.md",
        "dest": "reports/fused8_repeatability_comparison.md",
        "required": True,
        "kind": "file",
        "why": "repeatability comparison",
    },
    {
        "source": "runs/fused8_gemini_vs_template_comparison.md",
        "dest": "reports/fused8_gemini_vs_template_comparison.md",
        "required": False,
        "kind": "file",
        "why": "Gemini/template comparison if present",
    },
    {
        "source": "runs/fused8_all_model_comparison.md",
        "dest": "reports/fused8_all_model_comparison.md",
        "required": False,
        "kind": "file",
        "why": "all-model comparison if present",
    },
]


def package_artifacts(source_root: str | Path, out_archive: str | Path) -> Path:
    """Package selected RunPod research artifacts into a tar.gz archive."""

    source_root = Path(source_root)
    out_archive = Path(out_archive)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "artifacts": [],
        "missing": [],
        "omitted_secret_files": [],
    }
    with tempfile.TemporaryDirectory(prefix="okf_artifacts_") as tmp:
        staging = Path(tmp) / "package"
        staging.mkdir()
        for spec in ARTIFACT_SPECS:
            source = source_root / spec["source"]
            dest = staging / spec["dest"]
            entry = {
                "source": str(source),
                "dest": spec["dest"],
                "required": spec["required"],
                "kind": spec["kind"],
                "why": spec["why"],
                "present": source.exists(),
            }
            if not source.exists():
                manifest["missing"].append(entry)
                manifest["artifacts"].append(entry)
                continue
            if spec["kind"] == "run":
                copied = _copy_run_dir(source, dest, manifest)
            elif spec["kind"] == "dataset":
                copied = _copy_dataset_dir(source, dest, manifest)
            else:
                copied = _copy_file(source, dest, manifest)
            entry["copied_files"] = copied
            manifest["artifacts"].append(entry)

        manifest_path = staging / "artifact_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        checksums = _write_checksums(staging)
        manifest["checksums_file"] = "SHA256SUMS"
        manifest["file_count"] = len(checksums)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        _write_checksums(staging)

        out_archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out_archive, "w:gz") as tar:
            for path in sorted(staging.rglob("*")):
                tar.add(path, arcname=path.relative_to(staging))
    return out_archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package OpenKernelForge RunPod fused8 artifacts.")
    parser.add_argument("--source-root", default="/workspace/openkernelforge")
    parser.add_argument("--out", default="openkernelforge_fused8_artifacts.tar.gz")
    args = parser.parse_args(argv)
    archive = package_artifacts(args.source_root, args.out)
    print(f"Packaged artifacts: {archive}")
    return 0


def _copy_run_dir(source: Path, dest: Path, manifest: dict[str, Any]) -> int:
    copied = 0
    dest.mkdir(parents=True, exist_ok=True)
    for file_name in RUN_ROOT_FILES:
        path = source / file_name
        if path.exists() and path.is_file():
            copied += _copy_file(path, dest / file_name, manifest)
    for dir_name in RUN_DIRS:
        path = source / dir_name
        if path.exists() and path.is_dir():
            copied += _copy_tree(path, dest / dir_name, manifest)
    return copied


def _copy_dataset_dir(source: Path, dest: Path, manifest: dict[str, Any]) -> int:
    copied = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file() or _excluded(path):
            continue
        if path.suffix not in DATASET_SUFFIXES:
            continue
        copied += _copy_file(path, dest / path.relative_to(source), manifest)
    return copied


def _copy_tree(source: Path, dest: Path, manifest: dict[str, Any]) -> int:
    copied = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file() or _excluded(path):
            continue
        copied += _copy_file(path, dest / path.relative_to(source), manifest)
    return copied


def _copy_file(source: Path, dest: Path, manifest: dict[str, Any]) -> int:
    if source.is_symlink():
        raise ValueError(f"refusing to package symlinked artifact: {source}")
    if _contains_secret(source):
        manifest["omitted_secret_files"].append(str(source))
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return 1


def _write_checksums(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        rel = path.relative_to(root).as_posix()
        checksums[rel] = _sha256(path)
    lines = [f"{digest}  {rel}" for rel, digest in sorted(checksums.items())]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksums


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_secret(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as source:
            carry = ""
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    return any(pattern.search(carry) for pattern in SECRET_PATTERNS)
                text = carry + chunk
                if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                    return True
                carry = text[-512:]
    except OSError:
        return False


def _excluded(path: Path) -> bool:
    return any(part in EXCLUDED_NAMES for part in path.parts)


if __name__ == "__main__":
    raise SystemExit(main())
