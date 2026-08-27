import hashlib
from pathlib import Path

import pytest

from scripts.restore_frozen_multiplicity_candidates import restore_frozen_candidates


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_restore_requires_exact_frozen_hashes(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    output = tmp_path / "output"
    (staged / "task").mkdir(parents=True)
    source = staged / "task" / "template_00.py"
    metadata = staged / "task" / "template_00.json"
    source.write_text("def forward():\n    return 1\n", encoding="utf-8")
    metadata.write_text("{}\n", encoding="utf-8")
    row = {
        "candidate_id": "template_00",
        "sha256": _sha(source),
        "metadata_sha256": _sha(metadata),
    }
    payload = {"tasks": {"task": [row]}}

    restore_frozen_candidates(
        frozen=payload,
        regenerated=payload,
        staged_root=staged,
        output_root=output,
    )

    assert _sha(output / "task" / "template_00.py") == row["sha256"]


def test_restore_rejects_mismatched_source(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    (staged / "task").mkdir(parents=True)
    source = staged / "task" / "template_00.py"
    metadata = staged / "task" / "template_00.json"
    source.write_text("bad\n", encoding="utf-8")
    metadata.write_text("{}\n", encoding="utf-8")
    regenerated = {
        "tasks": {
            "task": [
                {
                    "candidate_id": "template_00",
                    "sha256": _sha(source),
                    "metadata_sha256": _sha(metadata),
                }
            ]
        }
    }
    frozen = {
        "tasks": {
            "task": [
                {
                    "candidate_id": "template_00",
                    "sha256": "0" * 64,
                    "metadata_sha256": _sha(metadata),
                }
            ]
        }
    }

    with pytest.raises(RuntimeError, match="source checksum mismatch"):
        restore_frozen_candidates(
            frozen=frozen,
            regenerated=regenerated,
            staged_root=staged,
            output_root=tmp_path / "output",
        )
