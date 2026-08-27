from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OFFICIAL_KERNELBENCH_URL = "https://github.com/ScalingIntelligence/KernelBench.git"
OFFICIAL_KERNELBENCH_COMMIT = "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"
DEFAULT_SMOKE_CONFIG = ROOT / "configs" / "kernelbench_l1_5task_corrected_rigorous.yaml"
DEFAULT_PILOT_CONFIG = ROOT / "configs" / "kernelbench_l1_20task_corrected_rigorous_safe.yaml"
SOURCE_FINGERPRINT_ROOTS = ("openkernelforge", "configs", "scripts", "tests")
SOURCE_FINGERPRINT_FILES = ("README.md", "pyproject.toml")


class CampaignError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the corrected, baseline-only KernelBench L1 CUDA campaign. "
            "The 20-task stage is gated on a clean 5-task stage."
        )
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--kernelbench-dir", default="/workspace/KernelBench")
    parser.add_argument("--kernelbench-commit", default=OFFICIAL_KERNELBENCH_COMMIT)
    parser.add_argument("--smoke-config", default=str(DEFAULT_SMOKE_CONFIG))
    parser.add_argument("--pilot-config", default=str(DEFAULT_PILOT_CONFIG))
    parser.add_argument(
        "--artifact-root",
        default="artifacts/corrected_cuda_campaign",
        help="Campaign manifests are written here after CUDA preflight succeeds.",
    )
    parser.add_argument("--max-wall-hours", type=float, default=5.0)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument(
        "--clone-kernelbench-if-missing",
        action="store_true",
        help="Clone the official repository only after the CUDA preflight succeeds.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    kernelbench_dir = Path(args.kernelbench_dir).resolve()
    smoke_config = _resolve_path(args.smoke_config, repo_root)
    pilot_config = _resolve_path(args.pilot_config, repo_root)
    artifact_root = _resolve_path(args.artifact_root, repo_root)
    if args.max_wall_hours <= 0:
        raise CampaignError("--max-wall-hours must be positive")

    # This check intentionally happens before creating an artifact directory or
    # cloning KernelBench. A CPU machine cannot accidentally produce campaign data.
    preflight = _cuda_preflight(repo_root)
    _validate_baseline_only_config(smoke_config, expected_tasks=5)
    _validate_baseline_only_config(pilot_config, expected_tasks=20)

    if not kernelbench_dir.exists():
        if not args.clone_kernelbench_if_missing:
            raise CampaignError(
                f"KernelBench checkout missing: {kernelbench_dir}. Pass "
                "--clone-kernelbench-if-missing on the CUDA host."
            )
        _run_checked(
            ["git", "clone", OFFICIAL_KERNELBENCH_URL, str(kernelbench_dir)],
            cwd=repo_root,
            timeout_seconds=900,
        )
    kernelbench_commit = _require_kernelbench_commit(kernelbench_dir, args.kernelbench_commit)

    started_monotonic = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    campaign_dir = artifact_root / stamp
    campaign_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "campaign_kind": "corrected_kernelbench_l1_baseline_only",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "max_wall_hours": args.max_wall_hours,
        "repo_root": str(repo_root),
        "source_git": _git_state(repo_root),
        "source_fingerprint": _source_fingerprint(repo_root),
        "kernelbench": {
            "path": str(kernelbench_dir),
            "url": OFFICIAL_KERNELBENCH_URL,
            "required_commit": args.kernelbench_commit,
            "observed_commit": kernelbench_commit,
        },
        "preflight": preflight,
        "configs": {
            "smoke": _config_provenance(smoke_config),
            "pilot": _config_provenance(pilot_config),
        },
        "stages": [],
        "status": "running",
    }
    _write_json(campaign_dir / "campaign_manifest.json", manifest)

    try:
        if not args.skip_install:
            remaining = _remaining_seconds(started_monotonic, args.max_wall_hours)
            install = _run_command(
                [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", "."],
                cwd=repo_root,
                timeout_seconds=min(remaining, 1800),
            )
            manifest["install"] = install
            _write_json(campaign_dir / "campaign_manifest.json", manifest)
            if install["returncode"] != 0:
                raise CampaignError("editable package installation failed; see campaign manifest")

        smoke_stage = _run_stage(
            name="smoke_5task",
            expected_tasks=5,
            config_path=smoke_config,
            kernelbench_dir=kernelbench_dir,
            repo_root=repo_root,
            timeout_seconds=_remaining_seconds(started_monotonic, args.max_wall_hours),
        )
        manifest["stages"].append(smoke_stage)
        _write_json(campaign_dir / "campaign_manifest.json", manifest)
        if not smoke_stage["validation"]["passed"]:
            raise CampaignError("5-task gate failed; 20-task pilot was not started")

        pilot_stage = _run_stage(
            name="pilot_20task",
            expected_tasks=20,
            config_path=pilot_config,
            kernelbench_dir=kernelbench_dir,
            repo_root=repo_root,
            timeout_seconds=_remaining_seconds(started_monotonic, args.max_wall_hours),
        )
        manifest["stages"].append(pilot_stage)
        if not pilot_stage["validation"]["passed"]:
            raise CampaignError("20-task corrected baseline validation failed")
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = round(time.monotonic() - started_monotonic, 3)
        _write_json(campaign_dir / "campaign_manifest.json", manifest)
        _write_checksums(campaign_dir)

    print(f"Corrected CUDA campaign complete: {campaign_dir}")
    return 0


def _cuda_preflight(repo_root: Path) -> dict[str, Any]:
    if platform.system() == "Darwin":
        raise CampaignError("WRONG ENVIRONMENT: corrected CUDA campaign cannot run on Darwin")
    if not shutil.which("nvidia-smi"):
        raise CampaignError("WRONG ENVIRONMENT: nvidia-smi is unavailable")

    from openkernelforge.utils.env_probe import probe_environment

    env = probe_environment()
    if not env.cuda_available:
        raise CampaignError("WRONG ENVIRONMENT: CUDA is unavailable")
    if not env.triton_available:
        raise CampaignError("WRONG ENVIRONMENT: Triton is unavailable")
    if not env.tiny_triton_kernel_passed:
        raise CampaignError("WRONG ENVIRONMENT: tiny Triton kernel did not pass")
    smi = _run_command(
        ["nvidia-smi", "-q", "-d", "CLOCK,POWER,TEMPERATURE"],
        cwd=repo_root,
        timeout_seconds=60,
    )
    if smi["returncode"] != 0:
        raise CampaignError("nvidia-smi clock/power/temperature probe failed")
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "environment": env.to_dict(),
        "nvidia_smi": smi,
    }


def _validate_baseline_only_config(path: Path, *, expected_tasks: int) -> None:
    if not path.is_file():
        raise CampaignError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    kernelbench = raw.get("kernelbench") or {}
    benchmark = raw.get("benchmark") or {}
    execution = raw.get("execution") or {}
    errors: list[str] = []
    if int(kernelbench.get("max_tasks", 0)) != expected_tasks:
        errors.append(f"kernelbench.max_tasks must equal {expected_tasks}")
    if kernelbench.get("candidate_provider") != "none":
        errors.append("kernelbench.candidate_provider must be none")
    if benchmark.get("timing_mode") != "cuda_event":
        errors.append("benchmark.timing_mode must be cuda_event")
    if int(benchmark.get("independent_sessions", 0)) < 3:
        errors.append("benchmark.independent_sessions must be at least 3")
    if int(benchmark.get("repeat", 0)) < 100:
        errors.append("benchmark.repeat must be at least 100")
    if benchmark.get("include_torch_compile") is not True:
        errors.append("benchmark.include_torch_compile must be true")
    if benchmark.get("torch_compile_mode") != "max-autotune":
        errors.append("benchmark.torch_compile_mode must be max-autotune")
    if (benchmark.get("cache_flush") or {}).get("enabled") is not True:
        errors.append("benchmark.cache_flush.enabled must be true")
    for key in ("require_cuda", "require_triton", "require_tiny_triton_kernel"):
        if execution.get(key) is not True:
            errors.append(f"execution.{key} must be true")
    if errors:
        raise CampaignError(f"unsafe corrected campaign config {path}: " + "; ".join(errors))


def _validate_stage_data(data: dict[str, Any], *, expected_tasks: int) -> dict[str, Any]:
    errors: list[str] = []
    if data.get("status") != "completed":
        errors.append(f"status={data.get('status')!r}, expected 'completed'")
    if int(data.get("tasks_selected") or 0) != expected_tasks:
        errors.append(f"tasks_selected={data.get('tasks_selected')!r}, expected {expected_tasks}")
    if data.get("candidate_records"):
        errors.append("candidate records are present in a baseline-only campaign")
    if data.get("failures"):
        errors.append(f"run reports {len(data['failures'])} failure(s)")

    selected = [record for record in data.get("records") or [] if not record.get("skipped")]
    if len(selected) != expected_tasks:
        errors.append(f"non-skipped record count={len(selected)}, expected {expected_tasks}")
    eager_timed = 0
    compile_timed = 0
    cache_perturbed = 0
    for record in selected:
        task_id = record.get("task_id") or "unknown"
        summary = record.get("benchmark_summary") or {}
        if record.get("candidate_contract") != "model_new":
            errors.append(f"{task_id}: candidate_contract is not model_new")
        if record.get("reference_ok") is not True:
            errors.append(f"{task_id}: reference did not execute")
        if summary.get("benchmark_error"):
            errors.append(f"{task_id}: benchmark_error recorded")
        if summary.get("compile_error"):
            errors.append(f"{task_id}: compile_error recorded")
        if summary.get("eager_median_ms") is not None:
            eager_timed += 1
        if summary.get("torch_compile_median_ms") is not None:
            compile_timed += 1
        if summary.get("cache_flush_performed") is True:
            cache_perturbed += 1
        if summary.get("timing_mode") != "cuda_event":
            errors.append(f"{task_id}: timing mode is not cuda_event")
        if int(summary.get("independent_sessions") or 0) < 3:
            errors.append(f"{task_id}: fewer than 3 sessions")
        if int(summary.get("repeat") or 0) < 100:
            errors.append(f"{task_id}: fewer than 100 measured samples per session")
    if eager_timed != expected_tasks:
        errors.append(f"eager timing count={eager_timed}, expected {expected_tasks}")
    if compile_timed != expected_tasks:
        errors.append(f"torch.compile timing count={compile_timed}, expected {expected_tasks}")
    if cache_perturbed != expected_tasks:
        errors.append(f"cache perturbation count={cache_perturbed}, expected {expected_tasks}")
    return {
        "passed": not errors,
        "errors": errors,
        "expected_tasks": expected_tasks,
        "selected_tasks": len(selected),
        "eager_timed": eager_timed,
        "compile_timed": compile_timed,
        "cache_perturbed": cache_perturbed,
    }


def _run_stage(
    *,
    name: str,
    expected_tasks: int,
    config_path: Path,
    kernelbench_dir: Path,
    repo_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "openkernelforge.cli",
        "kernelbench-l1-check",
        "--config",
        str(config_path),
        "--kernelbench-dir",
        str(kernelbench_dir),
    ]
    started = datetime.now(timezone.utc).isoformat()
    result = _run_command(command, cwd=repo_root, timeout_seconds=timeout_seconds)
    report_path = _report_path_from_output(result["stdout"], repo_root)
    validation: dict[str, Any]
    data_path: Path | None = None
    if report_path is None:
        validation = {"passed": False, "errors": ["report path not found in command output"]}
    else:
        data_path = report_path.parent / "kernelbench_l1_check.json"
        if not data_path.is_file():
            validation = {"passed": False, "errors": [f"missing run JSON: {data_path}"]}
        else:
            data = json.loads(data_path.read_text(encoding="utf-8"))
            validation = _validate_stage_data(data, expected_tasks=expected_tasks)
    if result["returncode"] != 0:
        validation.setdefault("errors", []).append(
            f"command exited with status {result['returncode']}"
        )
        validation["passed"] = False
    return {
        "name": name,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "command": result,
        "report_path": str(report_path) if report_path else None,
        "data_path": str(data_path) if data_path else None,
        "validation": validation,
    }


def _report_path_from_output(stdout: str, repo_root: Path) -> Path | None:
    match = re.search(r"^KernelBench L1 check written:\s*(.+)$", stdout, flags=re.MULTILINE)
    if not match:
        return None
    return _resolve_path(match.group(1).strip(), repo_root)


def _require_kernelbench_commit(path: Path, expected: str) -> str:
    if not (path / ".git").exists():
        raise CampaignError(f"KernelBench path is not a git checkout: {path}")
    result = _run_checked(["git", "rev-parse", "HEAD"], cwd=path, timeout_seconds=60)
    observed = result["stdout"].strip()
    if observed != expected:
        raise CampaignError(
            f"KernelBench commit mismatch: observed {observed}, required {expected}. "
            "Use a separate checkout at the required commit."
        )
    return observed


def _remaining_seconds(started: float, max_hours: float) -> float:
    remaining = max_hours * 3600.0 - (time.monotonic() - started)
    if remaining <= 0:
        raise CampaignError("campaign wall-time budget exhausted")
    return remaining


def _run_checked(
    command: list[str], *, cwd: Path, timeout_seconds: float
) -> dict[str, Any]:
    result = _run_command(command, cwd=cwd, timeout_seconds=timeout_seconds)
    if result["returncode"] != 0:
        raise CampaignError(
            f"command failed ({result['returncode']}): {' '.join(command)}\n{result['stderr']}"
        )
    return result


def _run_command(
    command: list[str], *, cwd: Path, timeout_seconds: float
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=max(1.0, timeout_seconds),
            check=False,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        return {
            "argv": command,
            "cwd": str(cwd),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        raise CampaignError(
            f"command exceeded remaining campaign budget: {' '.join(command)}"
        ) from exc


def _config_provenance(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _git_state(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {"available": False}
    head = _run_command(["git", "rev-parse", "HEAD"], cwd=path, timeout_seconds=60)
    status = _run_command(["git", "status", "--porcelain"], cwd=path, timeout_seconds=60)
    return {
        "available": head["returncode"] == 0,
        "head": head["stdout"].strip() if head["returncode"] == 0 else None,
        "dirty": bool(status["stdout"].strip()) if status["returncode"] == 0 else None,
    }


def _source_fingerprint(root: Path) -> dict[str, Any]:
    files: list[Path] = []
    for name in SOURCE_FINGERPRINT_ROOTS:
        base = root / name
        if base.is_dir():
            files.extend(path for path in base.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    for name in SOURCE_FINGERPRINT_FILES:
        path = root / name
        if path.is_file():
            files.append(path)
    digest = hashlib.sha256()
    rows: list[dict[str, str]] = []
    for path in sorted(set(files)):
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        rows.append({"path": relative, "sha256": file_hash})
    return {"sha256": digest.hexdigest(), "file_count": len(rows), "files": rows}


def _write_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != checksum_path:
            lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | os.PathLike[str], root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignError as exc:
        print(f"CAMPAIGN BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
