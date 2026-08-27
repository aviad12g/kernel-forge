#!/usr/bin/env python3
"""Run null and known-slowdown evaluator controls in fresh CUDA processes."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import hashlib
from pathlib import Path
from typing import Any

import torch
import yaml

from openkernelforge.harness.paired_timing import (
    PairedTimingConfig,
    benchmark_paired_blocks,
    configure_precision_settings,
    measure_cuda_interval_ms,
    write_paired_timing_result,
)
from openkernelforge.tasks.base import KernelTask, TaskTolerance
from openkernelforge.utils.env_probe import TRITON_EXECUTION_OK, probe_environment


CONTROL_ROWS = 1024
CONTROL_FEATURES = 4096


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/workshop2026_holdout_protocol.yaml")
    parser.add_argument("--output-dir", default="artifacts/workshop2026/evaluator_controls")
    parser.add_argument("--processes", type=int)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--process-id", default="p00")
    parser.add_argument("--seed", type=int, default=71_101)
    args = parser.parse_args()
    _require_cuda_linux()
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8")) or {}
    control_config = protocol["controls"]["calibration"]
    processes = int(args.processes or control_config["processes"])
    blocks = int(args.blocks or control_config["blocks_per_process"])
    seed_base = int(args.seed_base or control_config["process_seed_base"])
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return _worker(root, process_id=args.process_id, seed=args.seed, blocks=blocks)

    paths: list[Path] = []
    for process_index in range(processes):
        process_id = f"p{process_index:02d}"
        output = root / f"{process_id}.json"
        completed = subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker",
                "--protocol",
                str(Path(args.protocol).resolve()),
                "--output-dir",
                str(root),
                "--process-id",
                process_id,
                "--seed",
                str(seed_base + process_index * 101),
                "--blocks",
                str(blocks),
            ]
        )
        if completed.returncode or not output.exists():
            raise RuntimeError(f"control worker failed: {process_id}")
        paths.append(output)
    _summarize(
        root,
        paths,
        expected_processes=processes,
        validity_config=protocol["campaign_validity"],
    )
    _write_sha256_manifest(root)
    print(f"evaluator controls: {root}")
    return 0


def _worker(root: Path, *, process_id: str, seed: int, blocks: int) -> int:
    precision = configure_precision_settings(
        allow_tf32_matmul=True,
        allow_tf32_cudnn=True,
        float32_matmul_precision="high",
    )
    task = _control_task()
    persistent_model = torch.nn.Linear(
        CONTROL_FEATURES,
        CONTROL_FEATURES,
        bias=False,
        device="cuda",
    ).eval()

    def persistent(x: torch.Tensor) -> torch.Tensor:
        return persistent_model(x)

    def identical_wrapper(x: torch.Tensor) -> torch.Tensor:
        return persistent(x)

    extra_rows, calibration_ratio = _calibrate_extra_work_rows(
        persistent,
        task,
        seed=seed,
    )

    def known_slowdown(x: torch.Tensor) -> torch.Tensor:
        output = persistent(x)
        persistent(x[:extra_rows])
        return output

    config = PairedTimingConfig(blocks=blocks, seed=seed)
    result = benchmark_paired_blocks(
        task,
        {
            "persistent": persistent,
            "identical_wrapper": identical_wrapper,
            "known_slowdown": known_slowdown,
        },
        process_id=process_id,
        config=config,
        device="cuda",
    )
    write_paired_timing_result(root / f"{process_id}.json", result)
    # Add the calibration after the dataclass serializer writes the core record.
    path = root / f"{process_id}.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["positive_control_extra_rows"] = extra_rows
    stored["positive_control_calibration_ratio"] = calibration_ratio
    stored["precision_settings"] = precision
    path.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")
    return 0


def _control_task() -> KernelTask:
    def generate(seed: int, shape, dtype, device):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        x = torch.randn(shape, generator=generator, dtype=torch.float32)
        return (x.to(device=device, dtype=dtype),)

    return KernelTask(
        task_id="evaluator_control_linear",
        name="Evaluator control linear",
        description=(
            "Synthetic persistent-lifecycle calibration control sized so a "
            "2-8% injected delay is resolvable above launch overhead."
        ),
        reference_fn=lambda x: x,
        input_generator=generate,
        allowed_dtypes=(torch.float32,),
        tolerance=TaskTolerance(rtol=0.0, atol=0.0),
        benchmark_shapes=[(CONTROL_ROWS, CONTROL_FEATURES)],
    )


def _calibrate_extra_work_rows(
    fn,
    task: KernelTask,
    *,
    seed: int,
) -> tuple[int, float]:
    inputs = task.generate_inputs(seed, task.benchmark_shapes[0], torch.float32, torch.device("cuda"))
    device = torch.device("cuda")
    with torch.no_grad():
        for _ in range(10):
            fn(*inputs)
        torch.cuda.synchronize(device)
    baseline_ms = statistics.median(
        measure_cuda_interval_ms(fn, inputs, launches=30, device=device)
        for _ in range(3)
    )
    maximum_rows = int(inputs[0].shape[0])

    def observed_ratio(extra_rows: int) -> float:
        def delayed(*call_inputs):
            output = fn(*call_inputs)
            fn(call_inputs[0][:extra_rows])
            return output

        delayed_ms = measure_cuda_interval_ms(
            delayed,
            inputs,
            launches=30,
            device=device,
        )
        return delayed_ms / baseline_ms

    return select_empirical_extra_rows(
        observed_ratio,
        maximum_rows=maximum_rows,
        target_ratio=1.04,
    )


def select_empirical_extra_rows(
    measure_ratio,
    *,
    maximum_rows: int,
    target_ratio: float,
    search_steps: int = 16,
) -> tuple[int, float]:
    """Calibrate discarded extra GEMM work against a target slowdown ratio."""

    if maximum_rows <= 0 or target_ratio <= 1.0 or search_steps <= 0:
        raise ValueError("invalid empirical extra-work calibration parameters")
    observations: dict[int, float] = {}

    def observe(rows: int) -> float:
        rows = max(1, min(maximum_rows, int(rows)))
        if rows not in observations:
            observations[rows] = float(measure_ratio(rows))
        return observations[rows]

    low = 1
    high = maximum_rows
    observe(low)
    observe(high)
    for _ in range(search_steps):
        if high - low <= 1:
            break
        middle = (low + high) // 2
        if observe(middle) < target_ratio:
            low = middle
        else:
            high = middle

    best_rows, best_ratio = min(
        observations.items(),
        key=lambda item: (abs(item[1] - target_ratio), item[0]),
    )
    return best_rows, best_ratio


def _summarize(
    root: Path,
    paths: list[Path],
    *,
    expected_processes: int,
    validity_config: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        process_id = data["process_id"]
        ratios: dict[str, list[float]] = {
            "null_wrapper_over_persistent": [],
            "positive_over_persistent": [],
        }
        for block in data["blocks"]:
            times = block["median_ms_per_launch"]
            base = float(times["persistent"])
            ratios["null_wrapper_over_persistent"].append(
                base / float(times["identical_wrapper"])
            )
            ratios["positive_over_persistent"].append(base / float(times["known_slowdown"]))
        for metric, values in ratios.items():
            rows.append(
                {
                    "process_id": process_id,
                    "metric": metric,
                    "median_ratio": statistics.median(values),
                    "blocks": len(values),
                }
            )
    csv_path = root / "evaluator_controls.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_metric: dict[str, list[float]] = {}
    for row in rows:
        by_metric.setdefault(str(row["metric"]), []).append(float(row["median_ratio"]))
    summary = {
        metric: {
            "processes": len(values),
            "median_process_ratio": statistics.median(values),
            "process_ratios": values,
        }
        for metric, values in by_metric.items()
    }
    (root / "evaluator_controls_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    validity = evaluate_calibration_validity(
        summary,
        expected_processes=expected_processes,
        validity_config=validity_config,
    )
    (root / "calibration_validity.json").write_text(
        json.dumps(validity, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Evaluator Controls",
        "",
        "These are evaluator-calibration controls, not generated-kernel benchmark results.",
        "",
    ]
    for metric, values in summary.items():
        lines.append(f"- `{metric}`: median process ratio {values['median_process_ratio']:.6f}")
    lines.extend(["", f"Formal calibration gate: **{validity['status']}**"])
    (root / "evaluator_controls.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_calibration_validity(
    summary: dict[str, dict[str, Any]],
    *,
    expected_processes: int,
    validity_config: dict[str, Any],
) -> dict[str, Any]:
    """Apply the prespecified null and known-slowdown calibration gates."""

    null = summary.get("null_wrapper_over_persistent", {})
    positive = summary.get("positive_over_persistent", {})
    null_ratio = _optional_float(null.get("median_process_ratio"))
    positive_ratio = _optional_float(positive.get("median_process_ratio"))
    null_bounds = [float(value) for value in validity_config["null_control"]["median_ratio_bounds"]]
    slowdown_bounds = (
        float(validity_config["known_slowdown_control"]["minimum_detected_slowdown"]),
        float(validity_config["known_slowdown_control"]["maximum_detected_slowdown"]),
    )
    practical_margin = float(validity_config["practical_speedup_margin"])
    detected_slowdown = None if positive_ratio in {None, 0.0} else 1.0 / positive_ratio - 1.0
    checks = {
        "exact_process_count": (
            int(null.get("processes", 0)) == expected_processes
            and int(positive.get("processes", 0)) == expected_processes
        ),
        "null_median_within_bounds": (
            null_ratio is not None and null_bounds[0] <= null_ratio <= null_bounds[1]
        ),
        "null_no_practical_promotion": (
            null_ratio is not None
            and null_ratio <= 1.0 + practical_margin
            and null_ratio >= 1.0 / (1.0 + practical_margin)
        ),
        "known_slowdown_detected_in_range": (
            detected_slowdown is not None
            and slowdown_bounds[0] <= detected_slowdown <= slowdown_bounds[1]
        ),
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "expected_processes": expected_processes,
        "null_median_ratio": null_ratio,
        "known_slowdown_ratio": positive_ratio,
        "detected_slowdown_fraction": detected_slowdown,
        "thresholds": {
            "null_median_ratio_bounds": null_bounds,
            "practical_speedup_margin": practical_margin,
            "detected_slowdown_bounds": list(slowdown_bounds),
        },
        "checks": checks,
    }


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _require_cuda_linux() -> None:
    environment = probe_environment()
    if platform.system() == "Darwin" or environment.viability != TRITON_EXECUTION_OK:
        raise RuntimeError("evaluator controls require a Linux CUDA/Triton environment")


def _write_sha256_manifest(root: Path) -> Path:
    output = root / "SHA256SUMS"
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    raise SystemExit(main())
