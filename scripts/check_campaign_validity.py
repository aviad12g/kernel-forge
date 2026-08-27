#!/usr/bin/env python3
"""Combine prespecified evaluator controls into a fail-closed campaign gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/workshop2026_holdout_protocol.yaml")
    parser.add_argument("--calibration-validity", required=True)
    parser.add_argument("--lifecycle-summary", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/workshop2026/campaign_validity.json",
    )
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    calibration_path = Path(args.calibration_validity).resolve()
    lifecycle_path = Path(args.lifecycle_summary).resolve()
    output_path = Path(args.output).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    payload = evaluate_campaign_validity(
        protocol=protocol,
        protocol_path=protocol_path,
        calibration=calibration,
        calibration_path=calibration_path,
        lifecycle=lifecycle,
        lifecycle_path=lifecycle_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"campaign validity: {payload['status']} ({output_path})")
    return 0 if payload["status"] == "PASS" else 1


def evaluate_campaign_validity(
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    calibration: dict[str, Any],
    calibration_path: Path,
    lifecycle: dict[str, Any],
    lifecycle_path: Path,
) -> dict[str, Any]:
    validity = protocol["campaign_validity"]
    lifecycle_config = protocol["controls"]["lifecycle"]
    expected_rows = int(lifecycle_config["processes"])
    expected_tasks = int(lifecycle.get("expected_tasks", 0))
    if expected_tasks > 0:
        expected_rows *= expected_tasks
    checks = {
        "calibration_passed": calibration.get("status") == "PASS",
        "lifecycle_passed": lifecycle.get("status") == "PASS",
        "lifecycle_complete_host_and_device_records": (
            not validity["lifecycle_control"]["require_complete_host_and_device_records"]
            or (
                expected_rows > 0
                and int(lifecycle.get("completed_process_rows", 0)) == expected_rows
                and lifecycle.get("median_host_lifecycle_inflation") is not None
                and lifecycle.get("median_enclosing_event_inflation") is not None
            )
        ),
    }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "study_id": protocol.get("study", {}).get("id"),
        "failed_control_policy": validity["failed_control_policy"],
        "checks": checks,
        "protocol": _provenance(protocol_path),
        "calibration": _provenance(calibration_path),
        "lifecycle": _provenance(lifecycle_path),
    }


def _provenance(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
