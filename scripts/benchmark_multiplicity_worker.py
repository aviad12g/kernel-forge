#!/usr/bin/env python3
"""CUDA worker that times every frozen multiplicity candidate independently."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from openkernelforge.harness.paired_timing import (
    PairedTimingConfig,
    benchmark_paired_blocks,
    configure_precision_settings,
)
from openkernelforge.harness.policy import check_candidate_policy
from openkernelforge.harness.runtime_policy import audit_candidate_runtime
from openkernelforge.harness.sandbox import load_candidate_from_path, unload_candidate
from openkernelforge.harness.verifier import verify_candidate
from openkernelforge.tasks.fused_tasks import get_fused_tasks
from openkernelforge.utils.env_probe import TRITON_EXECUTION_OK, probe_environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    job_path = Path(args.job).resolve()
    output_path = Path(args.output).resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    started = time.monotonic()
    output: dict[str, object] = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "job": job,
        "job_sha256": _sha256_file(job_path),
        "status": "started",
        "candidate_results": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        environment = probe_environment()
        output["environment"] = environment.to_dict()
        if environment.viability != TRITON_EXECUTION_OK:
            raise RuntimeError(f"multiplicity worker viability is {environment.viability}")
        protocol_path = Path(job["protocol_path"]).resolve()
        protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
        _validate_job(job, protocol=protocol, protocol_path=protocol_path)
        precision = protocol["environment"]["precision"]
        output["precision_settings"] = configure_precision_settings(
            allow_tf32_matmul=bool(precision["allow_tf32_matmul"]),
            allow_tf32_cudnn=bool(precision["allow_tf32_cudnn"]),
            float32_matmul_precision=str(precision["float32_matmul_precision"]),
        )
        tasks = {task.task_id: task for task in get_fused_tasks()}
        task = tasks[str(job["task_id"])]
        phase_config = protocol[str(job["phase"])]
        confirmation = protocol["confirmation"]
        timing = PairedTimingConfig(
            blocks=int(
                phase_config.get(
                    "paired_blocks",
                    phase_config.get("paired_blocks_per_process"),
                )
            ),
            warmup_launches=int(confirmation["warmup_launches"]),
            minimum_interval_ms=float(confirmation["minimum_interval_ms"]),
            seed=int(job["seed"]),
            cache_l2_multiplier=float(
                confirmation["cache_state_perturbation"]["l2_multiplier"]
            ),
            cache_minimum_size_mb=int(
                confirmation["cache_state_perturbation"]["minimum_size_mb"]
            ),
            cache_maximum_size_mb=int(
                confirmation["cache_state_perturbation"]["maximum_size_mb"]
            ),
            cache_mode=str(confirmation["cache_state_perturbation"]["mode"]),
        )
        candidates = list(job["candidates"])
        random.Random(int(job["seed"])).shuffle(candidates)
        output["candidate_order"] = [item["candidate_id"] for item in candidates]
        for candidate in candidates:
            result = _evaluate_candidate(task, candidate, protocol=protocol, timing=timing, job=job)
            output["candidate_results"].append(result)
        output["status"] = "completed"
    except Exception:
        output["status"] = "failed"
        output["error"] = traceback.format_exc()
    output["elapsed_s"] = time.monotonic() - started
    output["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["status"] == "completed" else 1


def _evaluate_candidate(task, candidate, *, protocol, timing, job) -> dict[str, object]:
    result: dict[str, object] = {
        "candidate_id": candidate["candidate_id"],
        "candidate_path": candidate["path"],
        "candidate_sha256": candidate["sha256"],
        "status": "started",
    }
    path = _repo_path(candidate["path"])
    if _sha256_file(path) != candidate["sha256"]:
        result.update(status="INVALID", error="candidate checksum mismatch")
        return result
    source = path.read_text(encoding="utf-8")
    policy = check_candidate_policy(source, allow_torch_fallback=False, require_triton=True)
    result["policy"] = policy.__dict__
    if not policy.passed:
        result["status"] = "INVALID"
        return result
    loaded = None
    try:
        loaded = load_candidate_from_path(path, require_forward=True)
        assert loaded.forward is not None
        correctness = protocol["correctness"]
        verification = verify_candidate(
            task,
            loaded.forward,
            candidate_name=str(candidate["candidate_id"]),
            seeds=[int(seed) for seed in correctness["seeds"]],
            shapes=task.benchmark_shapes[:1],
            dtype=torch.float32,
            device="cuda",
            deterministic_repeats=int(correctness["same_input_repeat_executions"]),
            require_alias_contract=bool(correctness["require_alias_contract"]),
        )
        result["verification"] = {
            "passed": verification.passed,
            "elapsed_s": verification.elapsed_s,
            "error": verification.error,
            "cases": [case.__dict__ for case in verification.cases],
        }
        if not verification.passed:
            result["status"] = "INVALID"
            return result
        runtime = audit_candidate_runtime(
            task,
            loaded.forward,
            seed=int(correctness["seeds"][0]),
            dtype=torch.float32,
            device="cuda",
        )
        result["runtime_policy"] = runtime.__dict__
        if not runtime.passed:
            result["status"] = "INVALID"
            return result
        paired = benchmark_paired_blocks(
            task,
            {"eager": task.reference_fn, "candidate": loaded.forward},
            process_id=str(job["process_id"]),
            config=timing,
            device="cuda",
            dtype=torch.float32,
        )
        result["paired_timing"] = paired.__dict__
        result["status"] = "completed"
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
    finally:
        if loaded is not None:
            unload_candidate(loaded)
    return result


def _validate_job(
    job: dict[str, object],
    *,
    protocol: dict[str, object],
    protocol_path: Path,
) -> None:
    manifest_path = _repo_path(str(job["candidate_manifest_path"]))
    if _sha256_file(manifest_path) != str(job["candidate_manifest_sha256"]):
        raise RuntimeError("multiplicity candidate manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_config = protocol.get("candidates", {})
    allowed_statuses = (
        candidate_config.get("allowed_manifest_statuses", ["FROZEN_BEFORE_ANY_TIMING"])
        if isinstance(candidate_config, dict)
        else ["FROZEN_BEFORE_ANY_TIMING"]
    )
    if manifest.get("status") not in allowed_statuses:
        raise RuntimeError("multiplicity candidate manifest is not in an allowed frozen state")
    if manifest.get("protocol_sha256") != _sha256_file(protocol_path):
        raise RuntimeError("multiplicity protocol changed after candidate freeze")
    if str(job["task_id"]) not in manifest.get("tasks", {}):
        raise RuntimeError("task is absent from multiplicity candidate manifest")


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parents[1] / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
