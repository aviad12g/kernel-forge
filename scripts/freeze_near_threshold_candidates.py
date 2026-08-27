#!/usr/bin/env python3
"""Freeze deterministic near-threshold candidates before GPU calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
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
        default="configs/workshop2026_near_threshold_multiplicity_protocol.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/workshop2026/near_threshold_multiplicity/candidate_pool",
    )
    args = parser.parse_args()
    protocol_path = _repo_path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    output_root = _repo_path(args.output_root)
    manifest_path = _repo_path(protocol["artifacts"]["candidate_pool_manifest"])
    if manifest_path.exists():
        raise FileExistsError(f"near-threshold candidate pool is frozen: {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    payload = freeze_candidate_pool(
        protocol,
        protocol_path=protocol_path,
        output_root=output_root,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.with_suffix(".sha256").write_text(
        f"{_sha256_file(manifest_path)}  {manifest_path.name}\n",
        encoding="utf-8",
    )
    print(f"near-threshold candidate pool: {manifest_path}")
    return 0


def freeze_candidate_pool(
    protocol: dict[str, Any],
    *,
    protocol_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    candidate_config = protocol["candidates"]
    prior = candidate_config["historical_design_prior"]
    prior_protocol_path = _repo_path(prior["protocol"])
    prior_manifest_path = _repo_path(prior["manifest"])
    prior_timing_path = _repo_path(prior["timing_blocks"])
    prior_protocol = yaml.safe_load(prior_protocol_path.read_text(encoding="utf-8")) or {}
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    historical_speedups = _historical_candidate_speedups(
        prior_timing_path,
        phase=str(prior["phase"]),
    )
    tasks = {task.task_id: task for task in get_fused_tasks()}
    agent = TemplateAgent(
        template_family="fused8",
        template_variants=dict(prior_protocol["candidates"]["template_variants"]),
    )
    expected_count = int(candidate_config["variants_per_task"])
    block_size = int(candidate_config["delay_kernel"]["block_size"])
    num_warps = int(candidate_config["delay_kernel"]["num_warps"])
    records: dict[str, list[dict[str, Any]]] = {}
    design_priors: dict[str, dict[str, Any]] = {}

    for task_id in [str(item) for item in protocol["tasks"]["ids"]]:
        if task_id not in tasks:
            raise ValueError(f"unknown fused8 task: {task_id}")
        base_id = str(candidate_config["base_candidates"][task_id])
        base_index = int(base_id.rsplit("_", 1)[1])
        generated = agent.generate_all(tasks[task_id])
        if base_index >= len(generated):
            raise ValueError(f"base candidate {base_id} is unavailable for {task_id}")
        base = generated[base_index]
        prior_record = _manifest_candidate(prior_manifest, task_id, base_id)
        base_sha = _sha256_text(base.source)
        if base_sha != str(prior_record["sha256"]):
            raise RuntimeError(
                f"regenerated source does not match historical manifest: {task_id}/{base_id}"
            )
        old_speedup = historical_speedups.get((task_id, base_id))
        if old_speedup is None:
            raise RuntimeError(f"historical timing is missing for {task_id}/{base_id}")
        work_units = [float(value) for value in candidate_config["delay_work_units"][task_id]]
        if len(work_units) != expected_count:
            raise ValueError(
                f"{task_id} has {len(work_units)} delay levels; expected {expected_count}"
            )
        task_root = output_root / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        records[task_id] = []
        for index, work in enumerate(work_units):
            candidate_id = f"delay_{index:02d}"
            source = inject_discarded_copy_work(
                base.source,
                work_units=work,
                block_size=block_size,
                num_warps=num_warps,
            )
            source_path = task_root / f"{candidate_id}.py"
            metadata_path = task_root / f"{candidate_id}.json"
            source_path.write_text(source, encoding="utf-8")
            metadata = {
                "schema_version": 1,
                "task_id": task_id,
                "candidate_id": candidate_id,
                "candidate_source": "deterministic_template_with_discarded_copy_work",
                "base_candidate_id": base_id,
                "base_candidate_name": base.name,
                "base_candidate_sha256": base_sha,
                "historical_base_speedup": old_speedup,
                "delay_work_units": work,
                "delay_block_size": block_size,
                "delay_num_warps": num_warps,
                "calibration_role": "candidate_pool_only",
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            records[task_id].append(
                {
                    "candidate_id": candidate_id,
                    "candidate_name": f"{base.name}_delay_{work:.3f}",
                    "path": _repo_relative(source_path),
                    "sha256": _sha256_file(source_path),
                    "metadata_path": _repo_relative(metadata_path),
                    "metadata_sha256": _sha256_file(metadata_path),
                    "base_candidate_id": base_id,
                    "delay_work_units": work,
                }
            )
        design_priors[task_id] = {
            "base_candidate_id": base_id,
            "base_candidate_sha256": base_sha,
            "historical_confirmation_speedup": old_speedup,
            "historical_timing_artifact": _repo_relative(prior_timing_path),
            "note": "Historical timings informed only the predeclared delay grid.",
        }

    pilot_prior = _optional_pilot_calibration_prior(candidate_config)
    return {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_ANY_TIMING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": protocol["study"]["id"],
        "protocol_path": _repo_relative(protocol_path),
        "protocol_sha256": _sha256_file(protocol_path),
        "candidate_source": candidate_config["source"],
        "calibration_excluded_from_primary_analysis": True,
        "historical_design_priors": design_priors,
        "pilot_calibration_design_prior": pilot_prior,
        "tasks": records,
    }


def _optional_pilot_calibration_prior(candidate_config: dict[str, Any]) -> dict[str, Any] | None:
    prior = candidate_config.get("pilot_calibration_design_prior")
    if not prior:
        return None
    selection_path = _repo_path(prior["selection_csv"])
    protocol_path = _repo_path(prior["protocol"])
    if not selection_path.exists() or not protocol_path.exists():
        raise FileNotFoundError("pilot calibration design-prior artifacts are missing")
    return {
        "protocol_path": _repo_relative(protocol_path),
        "protocol_sha256": _sha256_file(protocol_path),
        "selection_csv": _repo_relative(selection_path),
        "selection_csv_sha256": _sha256_file(selection_path),
        "disposition": str(prior["disposition"]),
        "used_for_primary_analysis": False,
    }


def inject_discarded_copy_work(
    base_source: str,
    *,
    work_units: float,
    block_size: int,
    num_warps: int,
) -> str:
    """Add fixed discarded Triton copy work without changing task semantics."""

    if not math.isfinite(work_units) or work_units <= 0:
        raise ValueError("delay work units must be finite and positive")
    if block_size <= 0 or num_warps <= 0:
        raise ValueError("delay kernel launch parameters must be positive")
    full_passes = int(math.floor(work_units))
    tail_fraction = work_units - full_passes
    tail_numerator = int(round(tail_fraction * 1000))
    if tail_numerator == 1000:
        full_passes += 1
        tail_numerator = 0
    delay_kernel = f'''\n\n@triton.jit
def _okf_discarded_copy_kernel(x_ptr, scratch_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(scratch_ptr + offsets, values, mask=mask)
'''
    marker = "\n\n@triton.jit\n"
    if marker not in base_source:
        raise ValueError("base candidate has no Triton kernel insertion point")
    source = base_source.replace(marker, delay_kernel + marker, 1)
    forward_lines = source.splitlines()
    unpack_index = next(
        (
            index
            for index, line in enumerate(forward_lines)
            if line.startswith("    ") and line.strip().endswith("= args")
        ),
        None,
    )
    if unpack_index is None:
        raise ValueError("base candidate forward argument unpack was not found")
    injected = [
        "    # Fixed discarded work is preregistered and excluded from task semantics.",
        "    _okf_delay_scratch = torch.empty_like(x)",
        "    _okf_delay_n = x.numel()",
    ]
    full_launch = (
        "    _okf_discarded_copy_kernel[(triton.cdiv(_okf_delay_n, "
        f"{block_size}),)](x, _okf_delay_scratch, _okf_delay_n, "
        f"BLOCK_SIZE={block_size}, num_warps={num_warps})"
    )
    injected.extend(full_launch for _ in range(full_passes))
    if tail_numerator:
        injected.extend(
            [
                f"    _okf_delay_tail = (_okf_delay_n * {tail_numerator} + 999) // 1000",
                (
                    "    _okf_discarded_copy_kernel[(triton.cdiv(_okf_delay_tail, "
                    f"{block_size}),)](x, _okf_delay_scratch, _okf_delay_tail, "
                    f"BLOCK_SIZE={block_size}, num_warps={num_warps})"
                ),
            ]
        )
    forward_lines[unpack_index + 1 : unpack_index + 1] = injected
    return "\n".join(forward_lines) + "\n"


def _historical_candidate_speedups(
    path: Path,
    *,
    phase: str,
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("phase") != phase:
                continue
            key = (str(row["task_id"]), str(row["candidate_id"]))
            process_id = str(row["process_id"])
            ratio = float(row["eager_ms"]) / float(row["candidate_ms"])
            grouped.setdefault(key, {}).setdefault(process_id, []).append(math.log(ratio))
    result: dict[tuple[str, str], float] = {}
    for key, processes in grouped.items():
        process_medians = [statistics.median(values) for values in processes.values()]
        result[key] = math.exp(statistics.median(process_medians))
    return result


def _manifest_candidate(
    manifest: dict[str, Any],
    task_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    for record in manifest.get("tasks", {}).get(task_id, []):
        if record.get("candidate_id") == candidate_id:
            return record
    raise ValueError(f"historical manifest lacks {task_id}/{candidate_id}")


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
