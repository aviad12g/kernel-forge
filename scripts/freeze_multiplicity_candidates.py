#!/usr/bin/env python3
"""Freeze deterministic fused8 candidates before multiplicity timing."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from openkernelforge.tasks.fused_tasks import get_fused_tasks
from openkernelforge.templates.template_agent import TemplateAgent


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/workshop2026_multiplicity_protocol.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/workshop2026/multiplicity/candidates",
    )
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    output_root = Path(args.output_root).resolve()
    manifest_path = output_root.parent / "candidate_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"multiplicity candidate manifest is frozen: {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    payload = freeze_candidates(protocol, protocol_path=protocol_path, output_root=output_root)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum = manifest_path.with_suffix(".sha256")
    checksum.write_text(
        f"{_sha256_file(manifest_path)}  {manifest_path.name}\n",
        encoding="utf-8",
    )
    print(f"multiplicity candidate manifest: {manifest_path}")
    return 0


def freeze_candidates(
    protocol: dict[str, Any],
    *,
    protocol_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    selected_ids = [str(item) for item in protocol["tasks"]["ids"]]
    per_task = int(protocol["candidates"]["variants_per_task"])
    tasks = {task.task_id: task for task in get_fused_tasks()}
    missing = [task_id for task_id in selected_ids if task_id not in tasks]
    if missing:
        raise ValueError("unknown fused8 multiplicity tasks: " + ", ".join(missing))
    agent = TemplateAgent(
        template_family="fused8",
        template_variants=dict(protocol["candidates"]["template_variants"]),
    )
    records: dict[str, list[dict[str, Any]]] = {}
    for task_id in selected_ids:
        candidates = agent.generate_all(tasks[task_id])
        if len(candidates) != per_task:
            raise RuntimeError(
                f"{task_id} produced {len(candidates)} deterministic variants; expected {per_task}"
            )
        task_root = output_root / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        records[task_id] = []
        for index, candidate in enumerate(candidates):
            candidate_id = f"template_{index:02d}"
            source_path = task_root / f"{candidate_id}.py"
            metadata_path = task_root / f"{candidate_id}.json"
            source_path.write_text(candidate.source, encoding="utf-8")
            metadata = {
                "schema_version": 1,
                "task_id": task_id,
                "candidate_id": candidate_id,
                "candidate_name": candidate.name,
                "template_metadata": candidate.metadata,
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            records[task_id].append(
                {
                    "candidate_id": candidate_id,
                    "candidate_name": candidate.name,
                    "path": str(source_path.resolve()),
                    "sha256": _sha256_file(source_path),
                    "metadata_path": str(metadata_path.resolve()),
                    "metadata_sha256": _sha256_file(metadata_path),
                }
            )
    return {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_ANY_TIMING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": protocol["study"]["id"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256_file(protocol_path),
        "candidate_source": "deterministic_fused8_templates",
        "tasks": records,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
