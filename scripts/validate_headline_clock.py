from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from openkernelforge.harness.benchmarker import benchmark_task
from openkernelforge.harness.policy import CandidatePolicyResult, check_candidate_policy
from openkernelforge.harness.sandbox import load_candidate_from_path
from openkernelforge.tasks.kernelbench_l1 import (
    bind_kernelbench_candidate,
    load_kernelbench_l1_tasks,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "reports" / "tables" / "headline_clock_validation.csv"
REPORT = ROOT / "reports" / "headline_clock_validation.md"
ARTIFACT_DIR = ROOT / "artifacts" / "headline_clock_validation"
IMPORT_ROOT = ROOT / "artifacts" / "runpod_imports"
HISTORICAL_RUN_IDS = {"20260520_202314", "20260520_213128"}
CURRENT_POLICY_VERSION = CandidatePolicyResult(passed=False).policy_version


FUSED8_TARGETS = [
    {
        "task": "bias_relu",
        "source": "template",
        "old_speedup": "single 1.029x; repeat 0.976x",
        "old_label": "SINGLE_RUN_ONLY_WIN",
        "candidate_path": "not preserved",
        "notes": "Exact deterministic template candidate from runs/20260520_155839 is not present locally.",
    },
    {
        "task": "residual",
        "source": "OpenAI mini",
        "old_speedup": "1.074x",
        "old_label": "REPEAT_STABLE_WIN",
        "candidate_path": "not preserved",
        "notes": "Exact OpenAI mini candidate from runs/20260520_163607 is not present locally.",
    },
    {
        "task": "bias_gelu",
        "source": "template",
        "old_speedup": "1.485x",
        "old_label": "REPEAT_STABLE_WIN",
        "candidate_path": "not preserved",
        "notes": "Exact deterministic template candidate from runs/20260520_155839 is not present locally.",
    },
    {
        "task": "rmsnorm",
        "source": "template",
        "old_speedup": "1.452x",
        "old_label": "REPEAT_STABLE_WIN",
        "candidate_path": "not preserved",
        "notes": "Exact deterministic template candidate from runs/20260520_155839 is not present locally.",
    },
]


KERNELBENCH_TARGETS = [
    {
        "task": "CrossEntropyLoss",
        "task_id": "KernelBench__level1__95_CrossEntropyLoss",
        "run_id": "20260520_202314",
        "source": "Gemini one-shot",
        "old_speedup": "1.992x",
        "old_label": "REPEAT_STABLE_WIN",
    },
    {
        "task": "TripletMarginLoss",
        "task_id": "KernelBench__level1__99_TripletMarginLoss",
        "run_id": "20260520_202314",
        "source": "Gemini one-shot",
        "old_speedup": "4.176x",
        "old_label": "REPEAT_STABLE_WIN",
    },
    {
        "task": "KLDivLoss",
        "task_id": "KernelBench__level1__98_KLDivLoss",
        "run_id": "20260520_213128",
        "source": "Gemini repair1",
        "old_speedup": "1.843x",
        "old_label": "REPEAT_STABLE_WIN",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clock-recorded validation for headline candidates.")
    parser.add_argument("--kernelbench-dir", default=os.environ.get("KERNELBENCH_DIR") or "/workspace/KernelBench")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=120)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--cache-flush-mb", type=int, default=128)
    parser.add_argument("--try-lock-clocks", action="store_true")
    parser.add_argument(
        "--one-shot-run-dir",
        type=Path,
        help="Corrected KernelBench run directory containing CE and Triplet records.",
    )
    parser.add_argument(
        "--repair-run-dir",
        type=Path,
        help="Corrected KernelBench repair run directory containing the KLDiv record.",
    )
    parser.add_argument(
        "--allow-historical-debug",
        action="store_true",
        help=(
            "Permit policy-clean sources from the affected historical runs to execute for "
            "debugging only. Such rows are never current validation evidence."
        ),
    )
    args = parser.parse_args()

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    env_before = _env_summary()
    _write_json(ARTIFACT_DIR / "environment_before.json", env_before)
    if not env_before.get("cuda_available"):
        rows = [_unavailable_row(target, "CUDA unavailable") for target in FUSED8_TARGETS]
        rows.extend(_unavailable_row(target, "CUDA unavailable") for target in KERNELBENCH_TARGETS)
        _write_outputs(rows, env_before, {}, "not attempted: CUDA unavailable")
        return 0

    clock_mode, lock_info = _try_lock_clocks() if args.try_lock_clocks else ("clock-recorded", {"attempted": False})
    _write_json(ARTIFACT_DIR / "clock_lock_attempt.json", lock_info)

    rows: list[dict[str, Any]] = []
    for target in FUSED8_TARGETS:
        rows.append(_unavailable_row(target, "exact candidate artifact not preserved"))

    kernelbench_dir = Path(args.kernelbench_dir)
    if not kernelbench_dir.exists():
        rows.extend(_unavailable_row(target, f"KernelBench checkout missing at {kernelbench_dir}") for target in KERNELBENCH_TARGETS)
    else:
        for target in KERNELBENCH_TARGETS:
            run_dir = (
                args.repair_run_dir
                if target["task"] == "KLDivLoss" and args.repair_run_dir
                else args.one_shot_run_dir
                if target["task"] != "KLDivLoss" and args.one_shot_run_dir
                else IMPORT_ROOT / "runs" / target["run_id"]
            )
            run_dir = run_dir.expanduser().resolve()
            if run_dir.name in HISTORICAL_RUN_IDS and not args.allow_historical_debug:
                rows.append(
                    _unavailable_row(
                        target,
                        "historical adapter output blocked from validation; supply a corrected run directory",
                        evidence_status="historical_adapter_output_only",
                    )
                )
                continue
            rows.append(
                _validate_kernelbench_target(
                    target,
                    run_dir=run_dir,
                    kernelbench_dir=kernelbench_dir,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    sessions=args.sessions,
                    cache_flush_mb=args.cache_flush_mb,
                    clock_mode=clock_mode,
                )
            )

    env_after = _env_summary()
    _write_json(ARTIFACT_DIR / "environment_after.json", env_after)
    if lock_info.get("locked"):
        reset = _run(["nvidia-smi", "-rgc"])
        reset_mem = _run(["nvidia-smi", "-rmc"])
        _write_json(ARTIFACT_DIR / "clock_reset.json", {"graphics": reset, "memory": reset_mem})

    _write_outputs(rows, env_before, env_after, clock_mode)
    return 0


def _validate_kernelbench_target(
    target: dict[str, Any],
    *,
    run_dir: Path,
    kernelbench_dir: Path,
    warmup: int,
    repeats: int,
    sessions: int,
    cache_flush_mb: int,
    clock_mode: str,
) -> dict[str, Any]:
    row = _load_result_row(run_dir, target["task_id"])
    candidate_path = _mapped_path(row.get("candidate_path"), run_dir=run_dir)
    historical_run = run_dir.name in HISTORICAL_RUN_IDS
    shape = tuple(int(x) for x in ((row.get("benchmark") or {}).get("shape") or []))
    dtype = (row.get("benchmark") or {}).get("dtype") or "float32"
    before = _clock_query()
    try:
        if not candidate_path.exists():
            raise FileNotFoundError(f"candidate missing: {candidate_path}")
        if not shape:
            raise ValueError("benchmark shape missing from artifact")
        source = candidate_path.read_text(encoding="utf-8", errors="replace")
        policy = check_candidate_policy(
            source,
            allow_torch_fallback=False,
            require_triton=True,
        )
        if not policy.passed:
            raise ValueError(f"current policy rejected candidate: {policy.rejection_reason}")
        if not historical_run and not (
            row.get("policy_version") == CURRENT_POLICY_VERSION and row.get("candidate_contract")
        ):
            raise ValueError(
                "run record lacks corrected candidate-contract and "
                f"{CURRENT_POLICY_VERSION} policy metadata"
            )
        task = load_kernelbench_l1_tasks(kernelbench_dir, task_ids=[target["task_id"]], max_tasks=1)[0]
        candidate_module = load_candidate_from_path(candidate_path, require_forward=False).module
        candidate_forward = bind_kernelbench_candidate(
            task,
            candidate_module,
            dtype=getattr(torch, dtype),
            device="cuda",
        )
        result = benchmark_task(
            task,
            candidate_forward,
            candidate_name=f"headline_clock_{target['task']}",
            shape=shape,
            dtype=dtype,
            device="cuda",
            warmup=warmup,
            repeats=repeats,
            timing_mode="cuda_event",
            independent_sessions=sessions,
            cache_flush_config={"enabled": True, "size_mb": cache_flush_mb, "mode": "write"},
            bootstrap_ci_config={"enabled": True, "samples": 1000},
            separate_compile_time=True,
            stable_session_threshold=0.98,
            enable_torch_compile=False,
        )
        validation_label = _label(result.speedup_vs_eager, result.stable_above_eager)
        raw_path = ARTIFACT_DIR / f"{target['task']}_benchmark.json"
        _write_json(raw_path, asdict(result))
        after = _clock_query()
        debug_replay = historical_run and result.benchmark_error is None
        return {
            "task": target["task"],
            "source": target["source"],
            "candidate_path": str(candidate_path.relative_to(ROOT)) if _under_root(candidate_path) else str(candidate_path),
            "old_speedup": target["old_speedup"],
            "old_label": target["old_label"],
            "validation_speedup_median": _fmt(result.speedup_vs_eager),
            "validation_per_session_speedups": _session_speedups(result.session_summaries),
            "validation_label": validation_label,
            "clocks": clock_mode,
            "clock_state_before": _compact_clock(before),
            "clock_state_after": _compact_clock(after),
            "temperature_power_notes": _temperature_power(before, after),
            "label_changed": "yes" if validation_label != target["old_label"] else "no",
            "status": (
                "historical_debug_replay"
                if debug_replay
                else "validated"
                if result.benchmark_error is None
                else "benchmark_failed"
            ),
            "notes": (
                result.benchmark_error
                or (
                    "historical debug replay; invalid as performance evidence"
                    if historical_run
                    else "separate clock validation; original benchmark table unchanged"
                )
            ),
            "evidence_status": (
                "historical_debug_only" if historical_run else "corrected_clock_validation"
            ),
        }
    except Exception as exc:
        after = _clock_query()
        return {
            "task": target["task"],
            "source": target["source"],
            "candidate_path": str(candidate_path) if candidate_path else "not available",
            "old_speedup": target["old_speedup"],
            "old_label": target["old_label"],
            "validation_speedup_median": "not available",
            "validation_per_session_speedups": "not available",
            "validation_label": "INSUFFICIENT_DATA",
            "clocks": clock_mode,
            "clock_state_before": _compact_clock(before),
            "clock_state_after": _compact_clock(after),
            "temperature_power_notes": _temperature_power(before, after),
            "label_changed": "unknown",
            "status": "historical_debug_failed" if historical_run else "failed",
            "notes": f"{type(exc).__name__}: {exc}",
            "evidence_status": (
                "historical_debug_only" if historical_run else "corrected_clock_validation_failed"
            ),
        }


def _load_result_row(run_dir: Path, task_id: str) -> dict[str, Any]:
    candidates = [run_dir / "results.jsonl"]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_id") == task_id and row.get("verification_passed"):
                return row
    raise FileNotFoundError(f"verified row not found for {task_id} in {run_dir}")


def _mapped_path(path_text: str | None, *, run_dir: Path) -> Path:
    if not path_text:
        return Path("not available")
    path = Path(path_text)
    if path.is_absolute():
        return path
    for base in [run_dir, ROOT, IMPORT_ROOT]:
        candidate = base / path
        if candidate.exists():
            return candidate
    return ROOT / path


def _unavailable_row(
    target: dict[str, Any],
    reason: str,
    *,
    evidence_status: str = "unavailable",
) -> dict[str, Any]:
    return {
        "task": target["task"],
        "source": target["source"],
        "candidate_path": target.get("candidate_path", "not available"),
        "old_speedup": target["old_speedup"],
        "old_label": target["old_label"],
        "validation_speedup_median": "not available",
        "validation_per_session_speedups": "not available",
        "validation_label": "INSUFFICIENT_DATA",
        "clocks": "not run",
        "clock_state_before": "not available",
        "clock_state_after": "not available",
        "temperature_power_notes": "not available",
        "label_changed": "unknown",
        "status": "unavailable",
        "notes": reason if not target.get("notes") else f"{reason}; {target['notes']}",
        "evidence_status": evidence_status,
    }


def _label(speedup: float | None, stable_above_eager: bool | None) -> str:
    if speedup is None:
        return "INSUFFICIENT_DATA"
    if speedup >= 1.0 and stable_above_eager is True:
        return "REPEAT_STABLE_WIN"
    if speedup >= 1.0:
        return "UNSTABLE"
    return "BELOW_EAGER"


def _session_speedups(session_summaries: list[dict[str, Any]]) -> str:
    values = [summary.get("speedup_vs_eager") for summary in session_summaries]
    return ";".join(_fmt(value) for value in values if value is not None) or "not available"


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}x"
    except (TypeError, ValueError):
        return "not available"


def _try_lock_clocks() -> tuple[str, dict[str, Any]]:
    info: dict[str, Any] = {"attempted": True, "locked": False}
    current = _clock_query()
    supported = _run(["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"])
    info["supported_clocks"] = supported
    graphics = _int_or_none(current.get("graphics_clock_mhz"))
    memory = _int_or_none(current.get("memory_clock_mhz"))
    if graphics is None or memory is None or supported.get("returncode") != 0:
        info["reason"] = "clock values or supported-clock query unavailable"
        return "clock-recorded", info
    persistence = _run(["nvidia-smi", "-pm", "1"])
    # Lock to the current observed clocks only if the device accepts it. This is
    # conservative in the sense that it avoids selecting an unobserved boost state.
    lock_g = _run(["nvidia-smi", "-lgc", str(graphics)])
    lock_m = _run(["nvidia-smi", "-lmc", str(memory)])
    info.update({"persistence": persistence, "graphics_lock": lock_g, "memory_lock": lock_m})
    if lock_g.get("returncode") == 0 and lock_m.get("returncode") == 0:
        info["locked"] = True
        return "locked", info
    info["reason"] = "clock-lock command failed; validation remains clock-recorded"
    return "clock-recorded", info


def _env_summary() -> dict[str, Any]:
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": sys.platform,
    }
    try:
        data["torch"] = torch.__version__
        data["cuda_available"] = bool(torch.cuda.is_available())
        data["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        data["torch_error"] = f"{type(exc).__name__}: {exc}"
        data["cuda_available"] = False
    try:
        import triton

        data["triton"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:
        data["triton_error"] = f"{type(exc).__name__}: {exc}"
    data["nvidia_smi_clock_power_temperature"] = _run(["nvidia-smi", "-q", "-d", "CLOCK,POWER,TEMPERATURE"])
    data["clock_query"] = _clock_query()
    return data


def _clock_query() -> dict[str, str]:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,persistence_mode,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory",
            "--format=csv,noheader,nounits",
        ]
    )
    data = {"raw": result.get("stdout", "").strip(), "returncode": str(result.get("returncode"))}
    if result.get("returncode") == 0 and data["raw"]:
        parts = [part.strip() for part in data["raw"].split(",")]
        keys = ["gpu", "driver", "persistence", "temperature_c", "power_w", "graphics_clock_mhz", "memory_clock_mhz"]
        data.update({key: parts[index] for index, key in enumerate(keys) if index < len(parts)})
    return data


def _compact_clock(clock: dict[str, str]) -> str:
    if not clock:
        return "not available"
    return (
        f"gpu={clock.get('gpu', 'n/a')}; gfx={clock.get('graphics_clock_mhz', 'n/a')} MHz; "
        f"mem={clock.get('memory_clock_mhz', 'n/a')} MHz; temp={clock.get('temperature_c', 'n/a')} C; "
        f"power={clock.get('power_w', 'n/a')} W"
    )


def _temperature_power(before: dict[str, str], after: dict[str, str]) -> str:
    return (
        f"before temp/power={before.get('temperature_c', 'n/a')} C/{before.get('power_w', 'n/a')} W; "
        f"after temp/power={after.get('temperature_c', 'n/a')} C/{after.get('power_w', 'n/a')} W"
    )


def _run(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": -1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _under_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def _write_outputs(rows: list[dict[str, Any]], env_before: dict[str, Any], env_after: dict[str, Any], clock_mode: str) -> None:
    fields = [
        "task",
        "source",
        "candidate_path",
        "old_speedup",
        "old_label",
        "validation_speedup_median",
        "validation_per_session_speedups",
        "validation_label",
        "clocks",
        "clock_state_before",
        "clock_state_after",
        "temperature_power_notes",
        "label_changed",
        "status",
        "notes",
        "evidence_status",
    ]
    with TABLE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(ARTIFACT_DIR / "headline_clock_validation_rows.json", rows)
    validated = [
        row
        for row in rows
        if row["status"] == "validated"
        and row["evidence_status"] == "corrected_clock_validation"
    ]
    debug_replays = [
        row
        for row in rows
        if row["status"] == "historical_debug_replay"
        and row["evidence_status"] == "historical_debug_only"
    ]
    changed = [row for row in validated if row["label_changed"] == "yes"]
    lines = [
        "# Headline Clock Validation",
        "",
        "This validation rechecks selected headline candidates without generating candidates or overwriting original tables. Affected historical KernelBench runs are blocked by default; even explicit historical-debug rows are not paper evidence.",
        "",
        f"- Clock mode: {clock_mode}",
        f"- CUDA available: {env_before.get('cuda_available')}",
        f"- GPU: {env_before.get('gpu_name')}",
        f"- Torch: {env_before.get('torch')}",
        f"- Triton: {env_before.get('triton')}",
        f"- Corrected validation rows: {len(validated)}",
        f"- Historical debug replays: {len(debug_replays)}",
        f"- Label changes among corrected validation rows: {len(changed)}",
        "",
        "| Task | Source | Old label | Validation label | Validation speedup | Status | Notes |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['source']} | {row['old_label']} | {row['validation_label']} | "
            f"{row['validation_speedup_median']} | {row['status']} | {row['notes']} |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
