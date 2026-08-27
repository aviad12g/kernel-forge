#!/usr/bin/env python3
"""Generate a checksummed candidate manifest after task selection is frozen.

This command is intentionally separate from screening and timing. It makes one
model call per candidate, performs no correctness or performance evaluation,
and can resume only from complete, checksummed call artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from openkernelforge.agents.backends import create_backend
from openkernelforge.agents.code_extract import extract_python_code
from openkernelforge.config import AgentConfig
from openkernelforge.reports.kernelbench_l1 import build_kernelbench_prompt
from openkernelforge.tasks.kernelbench_l1 import load_kernelbench_l1_tasks


SYSTEM_MESSAGE = (
    "Generate one concise Python Triton implementation for the supplied official "
    "KernelBench task. Return only one Python code block."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/workshop2026_holdout_protocol.yaml")
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--kernelbench-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/workshop2026/candidates")
    parser.add_argument(
        "--allow-api-generation",
        action="store_true",
        help="Required acknowledgement that this command will make the prespecified API calls.",
    )
    args = parser.parse_args()
    if not args.allow_api_generation:
        raise RuntimeError("candidate generation is locked; pass --allow-api-generation")

    protocol_path = Path(args.protocol).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    generation = protocol.get("candidate_generation") or {}
    key_env = str(generation.get("api_key_env") or "")
    if not key_env or not os.environ.get(key_env):
        raise RuntimeError(f"candidate generation requires environment variable {key_env or '<unset>'}")

    task_manifest_path = Path(args.task_manifest).resolve()
    task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    _validate_frozen_task_manifest(
        task_manifest,
        task_manifest_path=task_manifest_path,
        protocol_path=protocol_path,
        kernelbench_dir=Path(args.kernelbench_dir).resolve(),
        protocol=protocol,
    )
    output_root = Path(args.output_dir).resolve()
    final_manifest_path = output_root.parent / "candidate_manifest.json"
    if final_manifest_path.exists():
        raise FileExistsError(f"candidate manifest is already frozen: {final_manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    backend = create_backend(
        AgentConfig(
            type="llm",
            backend=str(generation["backend"]),
            model=str(generation["configured_model_string"]),
            base_url=str(generation["base_url"]),
            api_key_env=key_env,
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            max_tokens=int(generation["max_tokens"]),
            timeout_seconds=float(generation["timeout_seconds"]),
            max_attempts=1,
            candidates_per_attempt=1,
            stop_after_first_correct=False,
            benchmark_all_correct=False,
            allow_torch_fallback=False,
        )
    )
    selected_ids = [str(item) for item in task_manifest["selected_task_ids"]]
    tasks = load_kernelbench_l1_tasks(
        args.kernelbench_dir,
        task_ids=selected_ids,
    )
    by_id = {task.task_id: task for task in tasks}
    missing = [task_id for task_id in selected_ids if task_id not in by_id]
    if missing:
        raise RuntimeError("frozen tasks failed to load: " + ", ".join(missing))

    per_task = int(generation["candidates_per_task"])
    if per_task != 3 or int(generation["max_attempts_per_candidate"]) != 1:
        raise RuntimeError("workshop protocol requires exactly three one-shot candidates per task")
    task_manifest_sha = _sha256_file(task_manifest_path)
    records: dict[str, list[dict[str, Any]]] = {}
    for task_id in selected_ids:
        task = by_id[task_id]
        records[task_id] = []
        for candidate_index in range(per_task):
            record = _generate_one(
                backend,
                task,
                candidate_index=candidate_index,
                output_root=output_root,
                generation=generation,
            )
            records[task_id].append(record)
            _write_partial_manifest(
                output_root.parent / "candidate_manifest.partial.json",
                task_manifest_sha=task_manifest_sha,
                generation=generation,
                records=records,
            )

    manifest = _manifest_payload(
        task_manifest_sha=task_manifest_sha,
        generation=generation,
        records=records,
    )
    if manifest["provider_response_model_fields_preserved"] is not True:
        raise RuntimeError(
            "provider response model fields were not preserved for every call; "
            "the final candidate manifest was not frozen"
        )
    final_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_path = final_manifest_path.with_suffix(".sha256")
    checksum_path.write_text(
        f"{_sha256_file(final_manifest_path)}  {final_manifest_path.name}\n",
        encoding="utf-8",
    )
    print(f"candidate manifest: {final_manifest_path}")
    print(f"checksum: {checksum_path}")
    return 0


def _generate_one(
    backend: Any,
    task: Any,
    *,
    candidate_index: int,
    output_root: Path,
    generation: dict[str, Any],
) -> dict[str, Any]:
    task_key = hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()[:12]
    candidate_id = f"candidate_{candidate_index:03d}"
    root = output_root / task_key / candidate_id
    root.mkdir(parents=True, exist_ok=True)
    prompt_path = root / "prompt.txt"
    response_path = root / "raw_response.txt"
    candidate_path = root / "candidate.py"
    metadata_path = root / "metadata.json"
    expected = (prompt_path, response_path, candidate_path, metadata_path)
    if any(path.exists() for path in expected):
        if not all(path.exists() for path in expected):
            raise RuntimeError(f"partial candidate artifacts require manual review: {root}")
        return _record_from_existing(
            task_id=task.task_id,
            candidate_id=candidate_id,
            prompt_path=prompt_path,
            response_path=response_path,
            candidate_path=candidate_path,
            metadata_path=metadata_path,
        )

    prompt = build_kernelbench_prompt(task)
    prompt_path.write_text(prompt, encoding="utf-8")
    raw_response = backend.generate(
        prompt,
        system=SYSTEM_MESSAGE,
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        max_tokens=int(generation["max_tokens"]),
    )
    response_path.write_text(raw_response, encoding="utf-8")
    extraction = extract_python_code(raw_response)
    source = extraction.code or (
        "# Candidate extraction failed. The preserved raw response is authoritative.\n"
    )
    candidate_path.write_text(source, encoding="utf-8")
    provider_metadata = dict(getattr(backend, "last_response_metadata", {}) or {})
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task.task_id,
        "task_source_path": task.metadata.get("source_path"),
        "task_source_sha256": _sha256_file(Path(str(task.metadata["source_path"]))),
        "candidate_id": candidate_id,
        "prompt_version": generation["prompt_version"],
        "configured_model": generation["configured_model_string"],
        "provider_metadata": provider_metadata,
        "extraction": {
            "ok": extraction.ok,
            "error": extraction.error,
            "metadata": extraction.metadata,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _record_from_existing(
        task_id=task.task_id,
        candidate_id=candidate_id,
        prompt_path=prompt_path,
        response_path=response_path,
        candidate_path=candidate_path,
        metadata_path=metadata_path,
    )


def _record_from_existing(
    *,
    task_id: str,
    candidate_id: str,
    prompt_path: Path,
    response_path: Path,
    candidate_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("task_id") != task_id or metadata.get("candidate_id") != candidate_id:
        raise RuntimeError(f"candidate metadata identity mismatch: {metadata_path}")
    provider_model = (metadata.get("provider_metadata") or {}).get("provider_response_model")
    return {
        "candidate_id": candidate_id,
        "path": str(candidate_path.resolve()),
        "sha256": _sha256_file(candidate_path),
        "prompt_path": str(prompt_path.resolve()),
        "prompt_sha256": _sha256_file(prompt_path),
        "raw_response_path": str(response_path.resolve()),
        "raw_response_sha256": _sha256_file(response_path),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": _sha256_file(metadata_path),
        "provider_response_model": provider_model or "not_returned",
    }


def _manifest_payload(
    *,
    task_manifest_sha: str,
    generation: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    provider_models = [
        candidate["provider_response_model"]
        for candidates in records.values()
        for candidate in candidates
    ]
    return {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_SCREENING",
        "task_selection_manifest_sha256": task_manifest_sha,
        "provider": generation["provider"],
        "configured_model_string": generation["configured_model_string"],
        "provider_response_model_fields_preserved": bool(provider_models)
        and all(value != "not_returned" for value in provider_models),
        "prompt_version": generation["prompt_version"],
        "candidates_per_task": generation["candidates_per_task"],
        "max_attempts_per_candidate": generation["max_attempts_per_candidate"],
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "max_tokens": generation["max_tokens"],
        "tasks": records,
    }


def _write_partial_manifest(
    path: Path,
    *,
    task_manifest_sha: str,
    generation: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
) -> None:
    payload = _manifest_payload(
        task_manifest_sha=task_manifest_sha,
        generation=generation,
        records=records,
    )
    payload["status"] = "PARTIAL_NOT_FOR_SCREENING"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_frozen_task_manifest(
    manifest: dict[str, Any],
    *,
    task_manifest_path: Path,
    protocol_path: Path,
    kernelbench_dir: Path,
    protocol: dict[str, Any],
) -> None:
    if manifest.get("status") != "FROZEN_BEFORE_CANDIDATE_PERFORMANCE":
        raise RuntimeError("task selection manifest is not frozen")
    selected = manifest.get("selected_task_ids")
    target = int((protocol.get("kernelbench") or {}).get("target_tasks", 50))
    if not isinstance(selected, list) or len(selected) != target:
        raise RuntimeError(f"frozen task manifest must contain exactly {target} selected tasks")
    if manifest.get("protocol_sha256") != _sha256_file(protocol_path):
        raise RuntimeError("task selection manifest protocol hash is stale")
    expected_commit = str((protocol.get("kernelbench") or {}).get("commit"))
    actual_commit = subprocess.run(
        ["git", "-C", str(kernelbench_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != expected_commit or manifest.get("kernelbench_commit") != expected_commit:
        raise RuntimeError(
            f"KernelBench commit mismatch: expected {expected_commit}, found {actual_commit}"
        )
    checksum_path = task_manifest_path.with_suffix(".sha256")
    if checksum_path.exists():
        recorded = checksum_path.read_text(encoding="utf-8").split()[0]
        if recorded != _sha256_file(task_manifest_path):
            raise RuntimeError("task selection manifest checksum file does not match")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
