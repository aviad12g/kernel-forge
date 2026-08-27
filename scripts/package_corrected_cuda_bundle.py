from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (
    "openkernelforge",
    "docs/methodology",
    "scripts",
    "tests",
    "paper/workshop2026",
)
SOURCE_FILES = (
    "README.md",
    "pyproject.toml",
    "configs/workshop2026_holdout_protocol.yaml",
    "configs/workshop2026_multiplicity_protocol.yaml",
    "reports/workshop2026_gpu_handoff.md",
)
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
EXCLUDED_NAMES = {
    ".DS_Store",
    "main.aux",
    "main.bbl",
    "main.blg",
    "main.fdb_latexmk",
    "main.fls",
    "main.log",
    "main.out",
    "main.pdf",
    "workshop2026_draft.pdf",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".code-workspace"}
SECRET_PATTERNS = (
    re.compile(rb"AIza[A-Za-z0-9_-]{20,}"),
    re.compile(rb"rpa_[A-Za-z0-9_-]{20,}"),
    re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(rb"(?:OPENAI|GEMINI|RUNPOD|VULTR)_API_KEY\s*=\s*[^<\s][^\s]*"),
)


def package_bundle(source_root: Path, output: Path) -> Path:
    source_root = source_root.resolve()
    output = output.resolve()
    selected = _selected_files(source_root)
    if not selected:
        raise ValueError(f"no source files selected under {source_root}")
    for path in selected:
        if path.is_symlink():
            raise ValueError(f"refusing to package symlink: {path}")
        match = _secret_match(path)
        if match:
            raise ValueError(f"possible secret in {path.relative_to(source_root)}: {match}")

    with tempfile.TemporaryDirectory(prefix="okf-cuda-bundle-") as tmp:
        staging = Path(tmp)
        manifest_files = []
        for source in selected:
            relative = source.relative_to(source_root)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            manifest_files.append(
                {
                    "path": relative.as_posix(),
                    "size_bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
            )
        manifest = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "workshop 2026 corrected holdout and controlled multiplicity campaign",
            "required_kernelbench_commit": "423217d9fda91e0c2d67e4a43bf62f96f6d104f1",
            "entrypoint": "follow reports/workshop2026_gpu_handoff.md in order",
            "workshop_entrypoints": [
                "python scripts/freeze_kernelbench_task_selection.py",
                "python scripts/freeze_multiplicity_candidates.py",
                "python scripts/run_workshop2026_shakedown.py",
                "python scripts/generate_workshop2026_candidates.py",
                "python scripts/run_evaluator_controls.py",
                "python scripts/run_lifecycle_ablation.py",
                "python scripts/check_campaign_validity.py",
                "python scripts/run_holdout_confirmation_campaign.py",
                "python scripts/run_multiplicity_campaign.py",
                "python scripts/make_workshop2026_results_figure.py",
            ],
            "file_count": len(manifest_files),
            "files": manifest_files,
        }
        (staging / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_checksums(staging)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=path.relative_to(staging).as_posix())
    return output


def _selected_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory in SOURCE_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if (
                not path.is_file()
                or path.name in EXCLUDED_NAMES
                or any(part in EXCLUDED_PARTS for part in path.parts)
            ):
                continue
            if path.suffix in EXCLUDED_SUFFIXES:
                continue
            files.append(path)
    for name in SOURCE_FILES:
        path = root / name
        if path.is_file():
            files.append(path)
    return sorted(set(files))


def _secret_match(path: Path) -> str | None:
    data = path.read_bytes()
    for pattern in SECRET_PATTERNS:
        match = pattern.search(data)
        if match:
            prefix = match.group(0).split(b"=", 1)[0][:32]
            return prefix.decode("ascii", errors="replace")
    return None


def _write_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    lines = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != checksum_path
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a secret-scanned corrected CUDA campaign bundle.")
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--out", default="artifacts/openkernelforge_workshop2026_gpu_bundle.tar.gz")
    args = parser.parse_args()
    output = package_bundle(Path(args.source_root), Path(args.out))
    print(f"Corrected CUDA bundle written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
