#!/usr/bin/env python3
"""Exercise the GPU evidence path on a task excluded from the frozen study."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from openkernelforge.harness.paired_timing import (
    PairedTimingConfig,
    benchmark_paired_blocks,
    configure_precision_settings,
    write_paired_timing_result,
)
from openkernelforge.harness.policy import check_candidate_policy
from openkernelforge.harness.runtime_policy import audit_candidate_runtime
from openkernelforge.harness.sandbox import load_candidate_from_path, unload_candidate
from openkernelforge.harness.verifier import verify_candidate
from openkernelforge.tasks.fused_tasks import get_fused_tasks
from openkernelforge.tasks.kernelbench_l1 import load_kernelbench_l1_tasks
from openkernelforge.templates.template_agent import TemplateAgent
from openkernelforge.utils.env_probe import TRITON_EXECUTION_OK, probe_environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/workshop2026_holdout_protocol.yaml")
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--kernelbench-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="artifacts/workshop2026/shakedown_excluded_task",
    )
    args = parser.parse_args()
    _require_cuda_linux()

    protocol_path = Path(args.protocol).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    manifest_path = Path(args.task_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kernelbench_dir = Path(args.kernelbench_dir).resolve()
    _validate_frozen_inputs(
        manifest,
        protocol_path=protocol_path,
        protocol=protocol,
        kernelbench_dir=kernelbench_dir,
    )
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    task_id = select_excluded_shakedown_task(manifest)
    task = load_kernelbench_l1_tasks(
        kernelbench_dir,
        task_ids=[task_id],
        max_tasks=1,
    )[0]
    _validate_loaded_task_source(task, manifest, task_id)
    precision = protocol["environment"]["precision"]
    precision_record = configure_precision_settings(
        allow_tf32_matmul=bool(precision["allow_tf32_matmul"]),
        allow_tf32_cudnn=bool(precision["allow_tf32_cudnn"]),
        float32_matmul_precision=str(precision["float32_matmul_precision"]),
    )

    def identical_reference_wrapper(*inputs: Any) -> Any:
        return task.reference_fn(*inputs)

    result = benchmark_paired_blocks(
        task,
        {"persistent_reference": task.reference_fn, "identical_wrapper": identical_reference_wrapper},
        process_id="excluded_shakedown",
        config=PairedTimingConfig(blocks=3, warmup_launches=3, seed=31_337),
        device="cuda",
        dtype=torch.float32,
    )
    timing_path = write_paired_timing_result(root / "paired_timing.json", result)
    candidate_gate = _run_triton_candidate_gate(root, protocol)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_eligibility": "EXCLUDED_SHAKEDOWN_NEVER_USE_FOR_PAPER_RESULTS",
        "task_id": task_id,
        "task_manifest_sha256": _sha256_file(manifest_path),
        "protocol_sha256": _sha256_file(protocol_path),
        "precision_settings": precision_record,
        "paired_timing_path": str(timing_path),
        "paired_timing_sha256": _sha256_file(timing_path),
        "candidate_gate": candidate_gate,
        "checks": {
            "task_excluded_from_frozen_selection": task_id
            not in set(manifest["selected_task_ids"]),
            "three_blocks_preserved": len(result.blocks) == 3,
            "input_hashes_preserved": all(
                bool(block.get("input_snapshot_sha256")) for block in result.blocks
            ),
            "telemetry_preserved": bool(result.environment),
            "cache_buffer_recorded": result.cache_buffer_mb > 0,
            "static_policy_passed": candidate_gate["policy_passed"],
            "correctness_passed": candidate_gate["verification_passed"],
            "runtime_policy_passed": candidate_gate["runtime_policy_passed"],
            "triton_launch_observed": candidate_gate["triton_launch_count"] >= 1,
        },
    }
    if not all(payload["checks"].values()):
        payload["status"] = "FAIL"
    summary_path = root / "shakedown_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"excluded-task shakedown: {payload['status']} ({summary_path})")
    return 0 if payload["status"] == "PASS" else 1


def _run_triton_candidate_gate(
    root: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Exercise policy, verifier, and runtime launch audit outside all studies."""

    task = next(task for task in get_fused_tasks() if task.task_id == "bias_relu")
    agent = TemplateAgent(
        template_family="fused8",
        template_variants={
            "block_sizes": [256],
            "num_warps": [4],
            "num_stages": [3],
            "contiguous_policies": ["none"],
            "output_allocation_policies": ["torch.empty"],
            "n_elements_modes": ["runtime"],
            "feature_dim_modes": ["runtime"],
            "max_variants_per_task": 1,
        },
    )
    candidates = agent.generate_all(task)
    if len(candidates) != 1:
        raise RuntimeError("shakedown expected exactly one deterministic Triton template")
    candidate_path = root / "excluded_triton_candidate.py"
    candidate_path.write_text(candidates[0].source, encoding="utf-8")
    policy = check_candidate_policy(
        candidates[0].source,
        allow_torch_fallback=False,
        require_triton=True,
    )
    if not policy.passed:
        raise RuntimeError(
            "deterministic shakedown candidate failed static policy: "
            f"{policy.rejection_reason}"
        )
    loaded = None
    try:
        loaded = load_candidate_from_path(candidate_path, require_forward=True)
        if loaded.forward is None:
            raise RuntimeError("shakedown candidate has no forward entry point")
        correctness = protocol["correctness"]
        verification = verify_candidate(
            task,
            loaded.forward,
            candidate_name="excluded_shakedown_template",
            seeds=[int(seed) for seed in correctness["seeds"]],
            shapes=task.benchmark_shapes[:1],
            dtype=torch.float32,
            device="cuda",
            deterministic_repeats=int(correctness["same_input_repeat_executions"]),
            require_alias_contract=bool(correctness["require_alias_contract"]),
        )
        runtime = audit_candidate_runtime(
            task,
            loaded.forward,
            seed=int(correctness["seeds"][0]),
            dtype=torch.float32,
            device="cuda",
        )
    finally:
        if loaded is not None:
            unload_candidate(loaded)
    return {
        "analysis_eligibility": "EXCLUDED_SHAKEDOWN_NEVER_USE_FOR_PAPER_RESULTS",
        "task_id": task.task_id,
        "candidate_path": str(candidate_path),
        "candidate_sha256": _sha256_file(candidate_path),
        "policy_passed": policy.passed,
        "policy_rejection_reason": policy.rejection_reason,
        "verification_passed": verification.passed,
        "verification_cases": [case.__dict__ for case in verification.cases],
        "runtime_policy_passed": runtime.passed,
        "triton_launch_count": runtime.triton_launch_count,
        "disallowed_aten_ops": runtime.disallowed_aten_ops,
        "runtime_error": runtime.error,
    }


def select_excluded_shakedown_task(manifest: dict[str, Any]) -> str:
    selected = {str(item) for item in manifest["selected_task_ids"]}
    candidates = [
        row
        for row in manifest.get("rows", [])
        if bool(row.get("feasible")) and str(row.get("task_id")) not in selected
    ]
    candidates.sort(
        key=lambda row: (str(row.get("source_relative_path")), str(row.get("task_id")))
    )
    if not candidates:
        raise RuntimeError("no feasible task outside the frozen study is available for shakedown")
    return str(candidates[0]["task_id"])


def _validate_frozen_inputs(
    manifest: dict[str, Any],
    *,
    protocol_path: Path,
    protocol: dict[str, Any],
    kernelbench_dir: Path,
) -> None:
    if manifest.get("status") != "FROZEN_BEFORE_CANDIDATE_PERFORMANCE":
        raise RuntimeError("shakedown requires the frozen task selection manifest")
    if manifest.get("protocol_sha256") != _sha256_file(protocol_path):
        raise RuntimeError("shakedown protocol differs from frozen task selection")
    expected_commit = str(protocol["kernelbench"]["commit"])
    actual_commit = subprocess.run(
        ["git", "-C", str(kernelbench_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if manifest.get("kernelbench_commit") != expected_commit or actual_commit != expected_commit:
        raise RuntimeError("shakedown KernelBench checkout differs from frozen protocol")


def _validate_loaded_task_source(
    task: Any,
    manifest: dict[str, Any],
    task_id: str,
) -> None:
    rows = [row for row in manifest.get("rows", []) if str(row.get("task_id")) == task_id]
    if len(rows) != 1:
        raise RuntimeError("shakedown task source row is missing or duplicated")
    if _sha256_file(Path(str(task.metadata["source_path"]))) != rows[0].get("source_sha256"):
        raise RuntimeError("shakedown task source differs from frozen manifest")


def _require_cuda_linux() -> None:
    environment = probe_environment()
    if platform.system() == "Darwin" or environment.viability != TRITON_EXECUTION_OK:
        raise RuntimeError("workshop shakedown requires a Linux CUDA/Triton environment")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
