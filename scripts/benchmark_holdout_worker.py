#!/usr/bin/env python3
"""Disposable CUDA worker for one task and one or more preserved candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
    cuda_environment_snapshot,
    hash_input_snapshot,
)
from openkernelforge.harness.policy import check_candidate_policy
from openkernelforge.harness.runtime_policy import audit_candidate_runtime
from openkernelforge.harness.sandbox import load_candidate_from_path, unload_candidate
from openkernelforge.harness.verifier import verify_candidate
from openkernelforge.tasks.kernelbench_l1 import (
    bind_kernelbench_candidate,
    load_kernelbench_l1_tasks,
)
from openkernelforge.utils.env_probe import TRITON_EXECUTION_OK, probe_environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    job_path = Path(args.job).resolve()
    output_path = Path(args.output).resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    worker_started = time.monotonic()
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
            raise RuntimeError(f"CUDA worker viability is {environment.viability}")
        protocol = yaml.safe_load(Path(job["protocol_path"]).read_text(encoding="utf-8"))
        precision = protocol["environment"]["precision"]
        output["precision_settings"] = configure_precision_settings(
            allow_tf32_matmul=bool(precision["allow_tf32_matmul"]),
            allow_tf32_cudnn=bool(precision["allow_tf32_cudnn"]),
            float32_matmul_precision=str(precision["float32_matmul_precision"]),
        )
        output["environment"] = cuda_environment_snapshot("cuda")
        _validate_frozen_task(job)
        task = load_kernelbench_l1_tasks(
            job["kernelbench_dir"],
            task_ids=[job["task_id"]],
            max_tasks=1,
        )[0]
        _validate_loaded_task_source(task, job)
        reference_init_args = getattr(task.reference_fn, "init_args", None)
        output["initialization_provenance"] = {
            "constructor_seed": task.metadata.get("model_init_seed"),
            "frozen_init_args_sha256": (
                hash_input_snapshot(reference_init_args)
                if reference_init_args is not None
                else None
            ),
            "semantics": (
                "reference and candidate use the same frozen constructor arguments "
                "and restored RNG seed; cross-module state equality is not required"
            ),
        }
        correctness = protocol["correctness"]
        timing_data = protocol[job["phase"]]
        confirmation = protocol["confirmation"]
        paired_config = PairedTimingConfig(
            blocks=int(
                timing_data.get("paired_blocks", timing_data.get("paired_blocks_per_process"))
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
        compiled_callable = None
        if job["phase"] == "screening" and timing_data.get("include_compile_baseline"):
            reference_callable = _prepare_callable(
                task.reference_fn,
                torch.float32,
                torch.device("cuda"),
            )
            compile_inputs = task.generate_inputs(
                int(job["seed"]),
                task.benchmark_shapes[0],
                torch.float32,
                torch.device("cuda"),
            )
            compile_started = time.perf_counter()
            compiled_callable = torch.compile(
                reference_callable,
                mode=str(timing_data.get("torch_compile_mode", "max-autotune")),
            )
            with torch.no_grad():
                compiled_callable(*compile_inputs)
                torch.cuda.synchronize()
            output["torch_compile"] = {
                "mode": timing_data.get("torch_compile_mode", "max-autotune"),
                "compile_and_first_call_ms": (time.perf_counter() - compile_started) * 1000.0,
                "materialized_before_runtime": True,
            }
        for candidate in job["candidates"]:
            candidate_result: dict[str, object] = {
                "candidate_id": candidate["candidate_id"],
                "candidate_path": candidate["path"],
                "candidate_sha256": candidate["sha256"],
                "status": "started",
            }
            output["candidate_results"].append(candidate_result)
            candidate_path = Path(candidate["path"]).resolve()
            if _sha256_file(candidate_path) != candidate["sha256"]:
                candidate_result["status"] = "INVALID"
                candidate_result["error"] = "candidate checksum mismatch"
                continue
            source = candidate_path.read_text(encoding="utf-8")
            policy = check_candidate_policy(
                source,
                allow_torch_fallback=False,
                require_triton=True,
            )
            candidate_result["policy"] = policy.__dict__
            if not policy.passed:
                candidate_result["status"] = "INVALID"
                continue
            loaded = None
            try:
                loaded = load_candidate_from_path(candidate_path, require_forward=False)
                candidate_callable = bind_kernelbench_candidate(
                    task,
                    loaded.module,
                    dtype=torch.float32,
                    device="cuda",
                )
                verification_lifecycle_before = _lifecycle_snapshot(
                    task.reference_fn,
                    candidate_callable,
                )
                verification = verify_candidate(
                    task,
                    candidate_callable,
                    candidate_name=candidate["candidate_id"],
                    seeds=[int(seed) for seed in correctness["seeds"]],
                    shapes=task.benchmark_shapes[:1],
                    dtype=torch.float32,
                    device="cuda",
                    deterministic_repeats=int(
                        correctness.get("same_input_repeat_executions", 1)
                    ),
                    require_alias_contract=bool(
                        correctness.get("require_alias_contract", False)
                    ),
                )
                verification_lifecycle_after = _lifecycle_snapshot(
                    task.reference_fn,
                    candidate_callable,
                )
                candidate_result["verification_lifecycle_invariant"] = {
                    "before": verification_lifecycle_before,
                    "after": verification_lifecycle_after,
                    "unchanged": verification_lifecycle_before
                    == verification_lifecycle_after,
                }
                candidate_result["verification"] = {
                    "passed": verification.passed,
                    "elapsed_s": verification.elapsed_s,
                    "error": verification.error,
                    "cases": [case.__dict__ for case in verification.cases],
                }
                if not verification.passed or (
                    verification_lifecycle_before != verification_lifecycle_after
                ):
                    candidate_result["status"] = "INVALID"
                    if verification_lifecycle_before != verification_lifecycle_after:
                        candidate_result["error"] = (
                            "reference or candidate state changed during correctness verification"
                        )
                    continue
                runtime_policy = audit_candidate_runtime(
                    task,
                    candidate_callable,
                    seed=int(correctness["seeds"][0]),
                    dtype=torch.float32,
                    device="cuda",
                )
                candidate_result["runtime_policy"] = runtime_policy.__dict__
                if not runtime_policy.passed:
                    candidate_result["status"] = "INVALID"
                    continue
                lifecycle_before = _lifecycle_snapshot(task.reference_fn, candidate_callable)
                methods = {"eager": task.reference_fn, "candidate": candidate_callable}
                if compiled_callable is not None:
                    methods["compile"] = compiled_callable
                paired = benchmark_paired_blocks(
                    task,
                    methods,
                    process_id=job["process_id"],
                    config=paired_config,
                    device="cuda",
                    dtype=torch.float32,
                )
                lifecycle_after = _lifecycle_snapshot(task.reference_fn, candidate_callable)
                candidate_result["lifecycle_invariant"] = {
                    "before": lifecycle_before,
                    "after": lifecycle_after,
                    "unchanged": lifecycle_before == lifecycle_after,
                }
                if lifecycle_before != lifecycle_after:
                    candidate_result["status"] = "INVALID"
                    candidate_result["error"] = (
                        "reference or candidate module lifecycle changed during timed blocks"
                    )
                    candidate_result["paired_timing"] = paired.__dict__
                    continue
                candidate_result["status"] = "completed"
                candidate_result["paired_timing"] = paired.__dict__
            except Exception:
                candidate_result["status"] = "failed"
                candidate_result["error"] = traceback.format_exc()
            finally:
                if loaded is not None:
                    unload_candidate(loaded)
        output["status"] = "completed"
    except Exception:
        output["status"] = "failed"
        output["error"] = traceback.format_exc()
    output["elapsed_s"] = time.monotonic() - worker_started
    output["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["status"] == "completed" else 1


def _validate_frozen_task(job: dict[str, object]) -> None:
    manifest_path = Path(str(job["task_manifest_path"])).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = str(job["task_manifest_sha256"])
    if _sha256_file(manifest_path) != expected_sha:
        raise RuntimeError("task manifest checksum mismatch")
    if manifest.get("status") != "FROZEN_BEFORE_CANDIDATE_PERFORMANCE":
        raise RuntimeError("task manifest is not frozen")
    protocol_path = Path(str(job["protocol_path"])).resolve()
    if manifest.get("protocol_sha256") != _sha256_file(protocol_path):
        raise RuntimeError("task manifest protocol hash mismatch")
    checkout = Path(str(job["kernelbench_dir"])).resolve()
    actual_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != manifest.get("kernelbench_commit"):
        raise RuntimeError("KernelBench checkout commit differs from frozen task manifest")
    if job["task_id"] not in manifest.get("selected_task_ids", []):
        raise RuntimeError(f"task is not in frozen selection: {job['task_id']}")


def _validate_loaded_task_source(task, job: dict[str, object]) -> None:
    manifest = json.loads(Path(str(job["task_manifest_path"])).read_text(encoding="utf-8"))
    rows = [row for row in manifest.get("rows", []) if row.get("task_id") == job["task_id"]]
    if len(rows) != 1:
        raise RuntimeError("frozen task source row is missing or duplicated")
    source_path = Path(str(task.metadata["source_path"])).resolve()
    if _sha256_file(source_path) != rows[0].get("source_sha256"):
        raise RuntimeError("loaded KernelBench task source differs from frozen manifest")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_callable(fn, dtype: torch.dtype, device: torch.device):
    prepare = getattr(fn, "prepare_for", None)
    if callable(prepare):
        prepared = prepare(dtype, device)
        if callable(prepared):
            return prepared
    return fn


def _lifecycle_snapshot(reference, candidate) -> dict[str, object]:
    return {
        "reference": _module_cache_snapshot(reference),
        "candidate": _module_cache_snapshot(candidate),
    }


def _module_cache_snapshot(value) -> list[dict[str, object]]:
    models = getattr(value, "_models", {})
    rows: list[dict[str, object]] = []
    for key, model in sorted(models.items(), key=lambda item: repr(item[0])):
        rows.append(
            {
                "cache_key": repr(key),
                "object_id": id(model),
                "state_sha256": _state_dict_sha256(model),
            }
        )
    return rows


def _state_dict_sha256(model) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
