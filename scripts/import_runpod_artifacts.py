from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def import_artifacts(archive: str | Path, out_dir: str | Path, *, force: bool = False) -> Path:
    """Safely extract a RunPod artifact archive and verify checksums."""

    archive = Path(archive)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(f"{out_dir} already exists and is not empty; pass --force to overwrite")
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        _validate_members(members)
        tar.extractall(out_dir)

    checksum_ok, checksum_errors = _verify_checksums(out_dir)
    report = _write_import_report(out_dir, archive, checksum_ok, checksum_errors)
    if not checksum_ok:
        raise RuntimeError(f"checksum validation failed; see {report}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import packaged OpenKernelForge RunPod artifacts.")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = import_artifacts(args.archive, args.out, force=args.force)
    except Exception as exc:
        print(f"Artifact import failed: {exc}")
        return 1
    print(f"Imported artifacts into {args.out}")
    print(f"Import report: {report}")
    return 0


def _validate_members(members: list[tarfile.TarInfo]) -> None:
    for member in members:
        name = member.name
        path = Path(name)
        if path.is_absolute():
            raise ValueError(f"unsafe absolute path in archive: {name}")
        if any(part == ".." for part in path.parts):
            raise ValueError(f"unsafe parent traversal in archive: {name}")
        if member.issym() or member.islnk():
            raise ValueError(f"archive links are not allowed: {name}")


def _verify_checksums(root: Path) -> tuple[bool, list[str]]:
    checksums = root / "SHA256SUMS"
    if not checksums.exists():
        return True, ["SHA256SUMS not present; skipped checksum verification"]
    errors: list[str] = []
    for line in checksums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        rel = rel.strip()
        path = root / rel
        if not path.exists():
            errors.append(f"missing checksummed file: {rel}")
            continue
        actual = _sha256(path)
        if actual != digest:
            errors.append(f"checksum mismatch: {rel}")
    return not errors, errors


def _write_import_report(root: Path, archive: Path, checksum_ok: bool, checksum_errors: list[str]) -> Path:
    report = root / "import_report.md"
    lines = [
        "# OpenKernelForge Artifact Import Report",
        "",
        f"- Archive: `{archive}`",
        f"- Output directory: `{root}`",
        f"- Checksum verification: {'PASS' if checksum_ok else 'FAIL'}",
        "",
        "## Checksum Details",
        "",
    ]
    if checksum_errors:
        lines.extend(f"- {error}" for error in checksum_errors)
    else:
        lines.append("- all checksums matched")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
