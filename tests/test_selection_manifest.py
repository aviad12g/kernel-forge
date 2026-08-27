from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from openkernelforge.tasks.selection_manifest import freeze_kernelbench_selection


def _make_checkout(root: Path, *, task_count: int = 6) -> str:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    level = root / "level1"
    level.mkdir()
    families = ["conv", "matmul", "pool", "loss", "relu", "sum"]
    for index in range(task_count):
        family = families[index % len(families)]
        (level / f"{index + 1}_{family}.py").write_text(
            "import torch\n"
            f"TASK_ID = 'task_{index + 1}'\n"
            "def reference_fn(x):\n    return x + 1\n"
            "def input_generator(seed, shape, dtype, device):\n"
            "    return (torch.empty((8, 8), dtype=dtype, device=device),)\n"
            "BENCHMARK_SHAPES = [(8, 8)]\n",
            encoding="utf-8",
        )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _write_protocol(path: Path, commit: str, target: int) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "study": {"id": "test"},
                "kernelbench": {
                    "commit": commit,
                    "target_tasks": target,
                    "family_order": [
                        "activation",
                        "convolution",
                        "loss",
                        "matmul",
                        "pooling",
                        "reduction",
                    ],
                    "selection_rule": "deterministic_first_feasible_family_round_robin",
                    "selection_blinding": "no_candidate_or_performance_fields_read",
                    "memory_preflight": {
                        "allow_cpu_materialization_fallback": True,
                        "max_estimated_known_peak_mb": 10,
                    },
                    "freeze": {
                        "manifest": "unused.json",
                        "csv": "unused.csv",
                        "checksum": "unused.sha256",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_selection_manifest_is_performance_blind_and_frozen(tmp_path: Path) -> None:
    checkout = tmp_path / "KernelBench"
    commit = _make_checkout(checkout)
    protocol = tmp_path / "protocol.yaml"
    _write_protocol(protocol, commit, 4)
    output = tmp_path / "out"
    paths = freeze_kernelbench_selection(protocol, checkout, output_root=output)
    data = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert data["status"] == "FROZEN_BEFORE_CANDIDATE_PERFORMANCE"
    assert len(data["selected_task_ids"]) == 4
    assert all(not row["candidate_or_performance_fields_read"] for row in data["rows"])
    assert paths.checksum.exists()
    with pytest.raises(FileExistsError):
        freeze_kernelbench_selection(protocol, checkout, output_root=output)


def test_selection_manifest_rejects_wrong_commit(tmp_path: Path) -> None:
    checkout = tmp_path / "KernelBench"
    _make_checkout(checkout)
    protocol = tmp_path / "protocol.yaml"
    _write_protocol(protocol, "0" * 40, 2)
    with pytest.raises(RuntimeError, match="commit mismatch"):
        freeze_kernelbench_selection(protocol, checkout, output_root=tmp_path / "out")
