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
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOTS = [ROOT, Path("/workspace/openkernelforge")]
TARGET_RUNS = [
    {
        "run_id": "20260520_155839",
        "description": "rigorous deterministic fused8 template run",
        "required_for": "fused8 uncertainty and full per-candidate flip recovery",
    },
    {
        "run_id": "20260520_163344",
        "description": "rigorous Gemini fused8 run",
        "required_for": "fused8 model uncertainty recovery",
    },
    {
        "run_id": "20260520_163607",
        "description": "rigorous OpenAI mini fused8 run",
        "required_for": "fused8 model uncertainty recovery",
    },
    {
        "run_id": "20260520_202314",
        "description": "KernelBench L1 capped Gemini one-shot pilot",
        "required_for": "KernelBench loss-win and failure interpretation",
    },
    {
        "run_id": "20260520_213128",
        "description": "KernelBench L1 capped Gemini repair pass",
        "required_for": "KernelBench repair interpretation",
    },
]
TARGET_DATASETS = [
    {
        "name": "fused8_curated_v1",
        "description": "curated fused8 dataset",
        "required_for": "dataset-level reproducibility",
    }
]
REPORT_FILES = [
    "runs/rigorous_fused8_model_comparison.md",
    "runs/kernelbench_gemini_repair1_comparison.md",
]
SECRET_PATTERNS = (
    re.compile(r"OPENAI_API_KEY\s*=\s*[^<\s][^\s]*"),
    re.compile(r"GEMINI_API_KEY\s*=\s*[^<\s][^\s]*"),
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"AIza[A-Za-z0-9_\-]{12,}"),
    re.compile(r"rpa_[A-Za-z0-9_\-]{20,}"),
)
EXCLUDED_NAMES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import already-existing OpenKernelForge RunPod artifacts into artifacts/runpod_imports."
    )
    parser.add_argument("--archive", help="Optional packaged tar.gz archive to extract safely.")
    parser.add_argument(
        "--source-root",
        action="append",
        dest="source_roots",
        help="Source root to search. May be passed multiple times.",
    )
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts" / "runpod_imports"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    if args.archive:
        manifest = import_artifacts(Path(args.archive).resolve(), out_dir)
    else:
        source_roots = [Path(p).resolve() for p in args.source_roots] if args.source_roots else DEFAULT_SOURCE_ROOTS
        manifest = import_artifacts(source_roots, out_dir)
    print(f"Wrote {out_dir / 'artifact_manifest.json'}")
    print(f"Wrote {out_dir / 'SHA256SUMS'}")
    copied = sum(1 for item in manifest["items"] if item["status"] == "copied")
    missing = sum(1 for item in manifest["items"] if item["status"] == "missing")
    print(f"Copied: {copied}; missing: {missing}")
    return 0


def import_artifacts(source_roots: list[Path] | Path | str, out_dir: Path | str) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if isinstance(source_roots, (str, Path)):
        source_path = Path(source_roots)
        if source_path.is_file():
            return _import_archive(source_path, out_dir)
        source_roots = [source_path]
    source_roots = [Path(path) for path in source_roots]
    return _import_from_roots(source_roots, out_dir)


def _import_from_roots(source_roots: list[Path], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_roots": [str(path) for path in source_roots],
        "target_root": str(out_dir),
        "items": [],
        "omitted_secret_files": [],
    }
    for spec in TARGET_RUNS:
        source = _find_first_existing(source_roots, Path("runs") / spec["run_id"])
        dest = out_dir / "runs" / spec["run_id"]
        _record_copy_or_missing(manifest, source, dest, "run", spec)
    for spec in TARGET_DATASETS:
        source = _find_first_existing(source_roots, Path("datasets") / spec["name"])
        dest = out_dir / "datasets" / spec["name"]
        _record_copy_or_missing(manifest, source, dest, "dataset", spec)
    for rel in REPORT_FILES:
        source = _find_first_existing(source_roots, Path(rel))
        dest = out_dir / rel
        _record_copy_or_missing(
            manifest,
            source,
            dest,
            "report",
            {"description": rel, "required_for": "paper traceability"},
        )
    manifest_path = out_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksums = _write_checksums(out_dir)
    manifest["checksums_file"] = "SHA256SUMS"
    manifest["sha256_count"] = len(checksums)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_checksums(out_dir)
    return manifest


def _import_archive(archive: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="okf-artifact-import-", dir=out_dir.parent) as temp_text:
        temp_root = Path(temp_text)
        with tarfile.open(archive, "r:gz") as tar:
            _extract_regular_members(tar, temp_root)
        verified_count = _verify_packaged_checksums(temp_root)
        secret_files = _secret_files(temp_root)
        if secret_files:
            raise ValueError(
                "archive contains possible secrets: "
                + ", ".join(str(path.relative_to(temp_root)) for path in secret_files[:5])
            )
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(temp_root, out_dir)
    manifest_path = out_dir / "artifact_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["imported_from_archive"] = str(archive)
    else:
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "imported_from_archive": str(archive),
            "items": [],
            "omitted_secret_files": [],
        }
    manifest["packaged_checksums_verified"] = verified_count
    (out_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_checksums(out_dir)
    return manifest


def _extract_regular_members(tar: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in tar.getmembers():
        target = (root / member.name).resolve()
        if not _is_within(target, root):
            raise ValueError(f"unsafe archive path: {member.name}")
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not member.isfile():
            raise ValueError(f"archive contains unsupported link or special file: {member.name}")
        source = tar.extractfile(member)
        if source is None:
            raise ValueError(f"could not read archive member: {member.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with source, target.open("wb") as output:
            shutil.copyfileobj(source, output)


def _verify_packaged_checksums(root: Path) -> int:
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.exists():
        raise ValueError("packaged artifact archive is missing SHA256SUMS")
    verified = 0
    listed: set[str] = set()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid SHA256SUMS row: {line!r}") from exc
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError(f"invalid SHA256 digest in row: {line!r}")
        if relative in listed:
            raise ValueError(f"duplicate SHA256SUMS path: {relative}")
        listed.add(relative)
        path = (root / relative).resolve()
        if not _is_within(path, root.resolve()) or not path.is_file():
            raise ValueError(f"checksum path missing or unsafe: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"checksum mismatch for {relative}")
        verified += 1
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    unlisted = sorted(actual_files.difference(listed))
    missing = sorted(listed.difference(actual_files))
    if unlisted:
        raise ValueError("archive contains unchecksummed files: " + ", ".join(unlisted[:5]))
    if missing:
        raise ValueError("SHA256SUMS lists missing files: " + ", ".join(missing[:5]))
    if not listed:
        raise ValueError("packaged artifact archive has an empty SHA256SUMS")
    return verified


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _record_copy_or_missing(
    manifest: dict[str, Any],
    source: Path | None,
    dest: Path,
    kind: str,
    spec: dict[str, str],
) -> None:
    entry: dict[str, Any] = {
        "kind": kind,
        "description": spec["description"],
        "required_for": spec["required_for"],
        "destination": str(dest),
    }
    if source is None:
        entry["status"] = "missing"
        entry["source"] = "not found in configured source roots"
        manifest["items"].append(entry)
        return
    if source.is_dir():
        copied_files = _copy_tree(source, dest, manifest)
    else:
        copied_files = _copy_file(source, dest, manifest)
    entry.update(
        {
            "status": "copied" if copied_files else "present_but_no_files_copied",
            "source": str(source),
            "copied_files": copied_files,
        }
    )
    manifest["items"].append(entry)


def _find_first_existing(source_roots: list[Path], rel: Path) -> Path | None:
    for root in source_roots:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def _copy_tree(source: Path, dest: Path, manifest: dict[str, Any]) -> int:
    copied = 0
    if dest.exists():
        shutil.rmtree(dest)
    for path in sorted(source.rglob("*")):
        if not path.is_file() or _excluded(path):
            continue
        copied += _copy_file(path, dest / path.relative_to(source), manifest)
    return copied


def _copy_file(source: Path, dest: Path, manifest: dict[str, Any]) -> int:
    if source.is_symlink():
        raise ValueError(f"refusing to copy symlinked artifact: {source}")
    if _contains_secret(source):
        manifest["omitted_secret_files"].append(str(source))
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return 1


def _excluded(path: Path) -> bool:
    return any(part in EXCLUDED_NAMES for part in path.parts)


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


def _secret_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS" and _contains_secret(path)
    ]


def _write_checksums(root: Path) -> list[str]:
    rows = []
    checksum_path = root / "SHA256SUMS"
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(root)}")
    checksum_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
