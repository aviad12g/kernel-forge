#!/usr/bin/env python3
"""Build a secret-scanned, checksummed workshop reviewer artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = (
    re.compile(rb"AIza[A-Za-z0-9_-]{20,}"),
    re.compile(rb"rpa_[A-Za-z0-9_-]{20,}"),
    re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(rb"(?:OPENAI|GEMINI|RUNPOD|VULTR)_API_KEY\s*=\s*[^<\s][^\s]*"),
)
EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "build",
    "build-upload",
    "build-submission",
}
EXCLUDED_SUFFIXES = {
    ".aux",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".pyc",
}


def package_release(source_root: Path, output: Path) -> Path:
    source_root = source_root.resolve()
    output = output.resolve()
    selected = _selected_files(source_root)
    if not selected:
        raise RuntimeError("no workshop release files selected")
    for path in selected:
        if path.is_symlink():
            raise RuntimeError(f"refusing to package symlink: {path}")
        match = find_secret_like_token(path)
        if match:
            raise RuntimeError(
                f"possible secret in {path.relative_to(source_root)}: {match}"
            )

    with tempfile.TemporaryDirectory(prefix="okf-workshop-release-") as temp:
        staging = Path(temp) / "openkernelforge_workshop2026_artifact"
        records = []
        for source in selected:
            relative = source.relative_to(source_root)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            records.append(
                {
                    "path": relative.as_posix(),
                    "size_bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
            )
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "NeurIPS 2026 ML for Systems workshop reviewer artifact",
            "kernelbench_commit": "423217d9fda91e0c2d67e4a43bf62f96f6d104f1",
            "claim_boundary": (
                "bounded evaluation-methodology evidence; no full KernelBench or SOTA claim"
            ),
            "file_count": len(records),
            "files": records,
        }
        (staging / "release_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _write_checksums(staging)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as archive:
            archive.add(staging, arcname=staging.name)
    return output


def find_secret_like_token(path: Path) -> str | None:
    data = path.read_bytes()
    for pattern in SECRET_PATTERNS:
        match = pattern.search(data)
        if match:
            token = match.group(0).split(b"=", 1)[0][:32]
            return token.decode("ascii", errors="replace")
    return None


def _selected_files(root: Path) -> list[Path]:
    selected: list[Path] = []
    artifact_root = root / "artifacts" / "workshop2026"
    if artifact_root.is_dir():
        selected.extend(_tree_files(artifact_root))

    for pattern in (
        "configs/workshop2026*.yaml",
        "reports/workshop2026*.md",
        "reports/tables/workshop2026*.csv",
    ):
        selected.extend(path for path in root.glob(pattern) if path.is_file())
    for relative in (
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "reports/artifact_index.md",
        "reports/reproducibility.md",
        "paper/workshop2026/README.md",
        "paper/workshop2026/openkernelforge_workshop2026.pdf",
        "paper/workshop2026/openkernelforge_workshop2026_submission.pdf",
    ):
        path = root / relative
        if path.is_file():
            selected.append(path)
    return sorted(set(selected))


def _tree_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in EXCLUDED_PARTS for part in path.parts)
        and path.suffix not in EXCLUDED_SUFFIXES
    ]


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument(
        "--out", default="dist/openkernelforge_workshop2026_artifacts.tar.gz"
    )
    args = parser.parse_args()
    output = package_release(Path(args.source_root), Path(args.out))
    print(f"workshop reviewer artifact: {output}")
    print(f"sha256: {_sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
