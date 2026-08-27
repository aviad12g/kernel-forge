from __future__ import annotations

import importlib.util
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_campaign_fails_closed_on_cpu_or_darwin() -> None:
    module = _load_script("run_holdout_confirmation_campaign.py")
    with pytest.raises(RuntimeError, match="Linux CUDA/Triton"):
        module._require_cuda_runpod()


def test_confirmation_waves_partition_seven_processes() -> None:
    module = _load_script("run_holdout_confirmation_campaign.py")
    assert module._confirmation_process_indices(7, wave="wave1", time_periods=2) == [0, 1, 2, 3]
    assert module._confirmation_process_indices(7, wave="wave2", time_periods=2) == [4, 5, 6]
    assert module._confirmation_process_indices(7, wave="all", time_periods=2) == list(range(7))
    assert [
        module._confirmation_wave_for_index(index, process_count=7, time_periods=2)
        for index in range(7)
    ] == ["wave1", "wave1", "wave1", "wave1", "wave2", "wave2", "wave2"]


def test_candidate_manifest_requires_complete_provenance(tmp_path: Path) -> None:
    module = _load_script("run_holdout_confirmation_campaign.py")
    artifacts = {}
    for field, name in {
        "path": "candidate.py",
        "prompt_path": "prompt.txt",
        "raw_response_path": "response.txt",
        "metadata_path": "metadata.json",
    }.items():
        path = tmp_path / name
        path.write_text(field, encoding="utf-8")
        artifacts[field] = str(path)
        artifacts[
            {
                "path": "sha256",
                "prompt_path": "prompt_sha256",
                "raw_response_path": "raw_response_sha256",
                "metadata_path": "metadata_sha256",
            }[field]
        ] = hashlib.sha256(path.read_bytes()).hexdigest()
    candidate = {
        "candidate_id": "candidate_000",
        "provider_response_model": "provider-model-id",
        **artifacts,
    }
    manifest = {
        "schema_version": 2,
        "status": "FROZEN_BEFORE_SCREENING",
        "task_selection_manifest_sha256": "task-sha",
        "provider_response_model_fields_preserved": True,
        "tasks": {"task": [candidate]},
    }
    module._validate_candidate_manifest(manifest, "task-sha")

    candidate["raw_response_sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="candidate provenance mismatch"):
        module._validate_candidate_manifest(manifest, "task-sha")


def test_task_manifest_requires_frozen_protocol_and_checkout_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("run_holdout_confirmation_campaign.py")
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text("kernelbench:\n  commit: kb-commit\n", encoding="utf-8")
    protocol = {"kernelbench": {"commit": "kb-commit"}}
    manifest = {
        "status": "FROZEN_BEFORE_CANDIDATE_PERFORMANCE",
        "protocol_sha256": module._sha256_file(protocol_path),
        "kernelbench_commit": "kb-commit",
    }

    class Completed:
        stdout = "kb-commit\n"

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Completed())
    module._validate_task_manifest(manifest, protocol_path, protocol, tmp_path)

    manifest["protocol_sha256"] = "stale"
    with pytest.raises(RuntimeError, match="protocol hash is stale"):
        module._validate_task_manifest(manifest, protocol_path, protocol, tmp_path)


def test_loaded_task_source_must_match_frozen_row(tmp_path: Path) -> None:
    module = _load_script("benchmark_holdout_worker.py")
    source = tmp_path / "task.py"
    source.write_text("class Model: pass\n", encoding="utf-8")
    manifest = {
        "rows": [
            {
                "task_id": "task-1",
                "source_sha256": module._sha256_file(source),
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class Task:
        metadata = {"source_path": str(source)}

    job = {"task_manifest_path": str(manifest_path), "task_id": "task-1"}
    module._validate_loaded_task_source(Task(), job)

    source.write_text("class Model: changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source differs"):
        module._validate_loaded_task_source(Task(), job)


def test_invalid_screening_tasks_are_preserved_instead_of_aborting(tmp_path: Path) -> None:
    module = _load_script("run_holdout_confirmation_campaign.py")
    task_id = "task_invalid"
    safe_task = hashlib.sha256(task_id.encode()).hexdigest()[:12]
    result_path = tmp_path / safe_task / "screening" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        '{"candidate_results":[{"candidate_id":"c0","status":"INVALID",'
        '"policy":{"passed":false,"rejection_reason":"missing_triton_kernel_launch"}}]}',
        encoding="utf-8",
    )
    invalid = module._screening_invalid_tasks(
        tmp_path,
        [task_id, "task_valid"],
        winner_task_ids={"task_valid"},
    )
    assert [item["task_id"] for item in invalid] == [task_id]
    rows = module._invalid_promotion_results(invalid, practical_margin=0.02)
    assert rows[0].label == "INVALID"
    assert rows[0].candidate_id == "none"


def test_screening_job_preserves_task_level_worker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("run_holdout_confirmation_campaign.py")
    calls = 0

    class Completed:
        returncode = 1

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": "torch.OutOfMemoryError: expected screening rejection",
                    "candidate_results": [],
                }
            ),
            encoding="utf-8",
        )
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    kwargs = {
        "root": tmp_path,
        "protocol_path": tmp_path / "protocol.yaml",
        "task_manifest_path": tmp_path / "tasks.json",
        "task_manifest_sha": "task-sha",
        "kernelbench_dir": tmp_path / "KernelBench",
        "phase": "screening",
        "task_id": "task_oom",
        "process_id": "screening",
        "seed": 1,
        "candidates": [],
        "allow_failed_result": True,
    }
    result = module._run_job(**kwargs)
    assert result["status"] == "failed"
    assert "OutOfMemoryError" in result["error"]

    # A resume consumes the preserved failure and does not execute the worker again.
    assert module._run_job(**kwargs) == result
    assert calls == 1


def test_confirmation_job_still_fails_closed_on_worker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("run_holdout_confirmation_campaign.py")

    class Completed:
        returncode = 1

    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_text(
            '{"status":"failed","error":"worker failed"}',
            encoding="utf-8",
        )
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="worker failed"):
        module._run_job(
            root=tmp_path,
            protocol_path=tmp_path / "protocol.yaml",
            task_manifest_path=tmp_path / "tasks.json",
            task_manifest_sha="task-sha",
            kernelbench_dir=tmp_path / "KernelBench",
            phase="confirmation",
            task_id="task",
            process_id="p00",
            seed=1,
            candidates=[],
        )


def test_recorded_worker_hours_and_checksum_ledger(tmp_path: Path) -> None:
    module = _load_script("run_holdout_confirmation_campaign.py")
    first = tmp_path / "screening" / "task" / "screening" / "result.json"
    second = tmp_path / "confirmation" / "task" / "p00" / "result.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text('{"elapsed_s": 1800}', encoding="utf-8")
    second.write_text('{"elapsed_s": 900}', encoding="utf-8")
    assert module._recorded_worker_hours(tmp_path) == pytest.approx(0.75)
    ledger = module._write_sha256_manifest(tmp_path)
    text = ledger.read_text(encoding="utf-8")
    assert "screening/task/screening/result.json" in text
    assert "SHA256SUMS" not in text


def test_campaign_gate_rejects_failed_or_stale_controls(tmp_path: Path) -> None:
    module = _load_script("run_holdout_confirmation_campaign.py")
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text("study:\n  id: study\n", encoding="utf-8")
    protocol = {"study": {"id": "study"}}
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "study_id": "study",
                "checks": {"calibration": True, "lifecycle": True},
                "protocol": {"sha256": module._sha256_file(protocol_path)},
            }
        ),
        encoding="utf-8",
    )
    module._validate_campaign_gate(gate_path, protocol_path, protocol)

    failed = json.loads(gate_path.read_text(encoding="utf-8"))
    failed["checks"]["lifecycle"] = False
    gate_path.write_text(json.dumps(failed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not pass"):
        module._validate_campaign_gate(gate_path, protocol_path, protocol)


def test_wave2_requires_elapsed_separation(tmp_path: Path) -> None:
    module = _load_script("run_holdout_confirmation_campaign.py")
    lock = {
        "wave2_not_before_utc": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
        "environment_fingerprint": {},
    }
    (tmp_path / "wave1_integrity_lock.json").write_text(
        json.dumps(lock),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="cannot start before"):
        module._validate_wave2_start(
            tmp_path,
            {
                "require_same_gpu_uuid": False,
                "require_same_software_fingerprint": False,
            },
        )
