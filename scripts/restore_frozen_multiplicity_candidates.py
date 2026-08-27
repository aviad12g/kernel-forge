#!/usr/bin/env python3
"""Restore omitted deterministic candidates only when frozen checksums match."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.freeze_multiplicity_candidates import freeze_candidates
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from freeze_multiplicity_candidates import freeze_candidates


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="configs/workshop2026_multiplicity_protocol.yaml"
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/workshop2026/multiplicity/candidate_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/workshop2026/multiplicity/candidates",
    )
    args = parser.parse_args()
    protocol_path = _resolve(args.protocol)
    manifest_path = _resolve(args.manifest)
    output_root = _resolve(args.output_root)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN_BEFORE_ANY_TIMING":
        raise RuntimeError("candidate manifest is not frozen")
    if frozen.get("protocol_sha256") != _sha256_file(protocol_path):
        raise RuntimeError("protocol checksum differs from the frozen manifest")

    with tempfile.TemporaryDirectory(prefix="okf-frozen-candidates-") as temp:
        staged_root = Path(temp) / "candidates"
        regenerated = freeze_candidates(
            protocol, protocol_path=protocol_path, output_root=staged_root
        )
        restore_frozen_candidates(
            frozen=frozen,
            regenerated=regenerated,
            staged_root=staged_root,
            output_root=output_root,
        )
    print(f"restored checksum-matched frozen candidates: {output_root}")
    return 0


def restore_frozen_candidates(
    *,
    frozen: dict[str, Any],
    regenerated: dict[str, Any],
    staged_root: Path,
    output_root: Path,
) -> None:
    expected = _indexed(frozen)
    actual = _indexed(regenerated)
    if set(expected) != set(actual):
        raise RuntimeError("regenerated candidate set differs from frozen manifest")
    for key in sorted(expected):
        if expected[key]["sha256"] != actual[key]["sha256"]:
            raise RuntimeError(f"source checksum mismatch for {key[0]}/{key[1]}")
        if expected[key]["metadata_sha256"] != actual[key]["metadata_sha256"]:
            raise RuntimeError(f"metadata checksum mismatch for {key[0]}/{key[1]}")

    for task_id, candidate_id in sorted(expected):
        task_root = output_root / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        for suffix in (".py", ".json"):
            source = staged_root / task_id / f"{candidate_id}{suffix}"
            destination = task_root / source.name
            if destination.exists() and _sha256_file(destination) != _sha256_file(source):
                raise RuntimeError(f"refusing to overwrite changed file: {destination}")
            shutil.copy2(source, destination)


def _indexed(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(task_id), str(row["candidate_id"])): row
        for task_id, rows in payload.get("tasks", {}).items()
        for row in rows
    }


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
