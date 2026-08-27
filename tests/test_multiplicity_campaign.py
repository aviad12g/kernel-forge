from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multiplicity_candidate_freeze_produces_twenty_per_task(tmp_path: Path) -> None:
    module = _load_script("freeze_multiplicity_candidates.py")
    protocol_path = ROOT / "configs" / "workshop2026_multiplicity_protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol["tasks"]["ids"] = ["bias_relu"]
    payload = module.freeze_candidates(
        protocol,
        protocol_path=protocol_path,
        output_root=tmp_path / "candidates",
    )
    assert payload["status"] == "FROZEN_BEFORE_ANY_TIMING"
    assert len(payload["tasks"]["bias_relu"]) == 20
    assert all(Path(row["path"]).is_file() for row in payload["tasks"]["bias_relu"])


def test_multiplicity_campaign_fails_closed_off_cuda() -> None:
    module = _load_script("run_multiplicity_campaign.py")
    with pytest.raises(RuntimeError, match="Linux CUDA/Triton"):
        module._require_cuda_linux()


def test_multiplicity_manifest_validation_checks_all_candidate_hashes(tmp_path: Path) -> None:
    module = _load_script("run_multiplicity_campaign.py")
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(
        "tasks: {ids: [task]}\ncandidates: {variants_per_task: 1}\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.py"
    metadata = tmp_path / "candidate.json"
    candidate.write_text("def forward(x): return x\n", encoding="utf-8")
    metadata.write_text("{}\n", encoding="utf-8")
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "status": "FROZEN_BEFORE_ANY_TIMING",
        "protocol_sha256": sha(protocol_path),
        "tasks": {
            "task": [
                {
                    "path": str(candidate),
                    "sha256": sha(candidate),
                    "metadata_path": str(metadata),
                    "metadata_sha256": sha(metadata),
                }
            ]
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    module._validate_manifest(
        manifest,
        manifest_path=manifest_path,
        protocol_path=protocol_path,
    )
    manifest["tasks"]["task"][0]["sha256"] = "bad"
    with pytest.raises(RuntimeError, match="source checksum mismatch"):
        module._validate_manifest(
            manifest,
            manifest_path=manifest_path,
            protocol_path=protocol_path,
        )
