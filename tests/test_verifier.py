from pathlib import Path

from openkernelforge.harness.sandbox import load_candidate_from_path
from openkernelforge.harness.verifier import verify_candidate
from openkernelforge.tasks.simple_tasks import get_task


def _load_tmp_candidate(tmp_path: Path, source: str):
    path = tmp_path / "candidate.py"
    path.write_text(source, encoding="utf-8")
    return load_candidate_from_path(path)


def test_verifier_passes_on_correct_torch_candidate(tmp_path):
    task = get_task("vector_add")
    candidate = _load_tmp_candidate(
        tmp_path,
        "def forward(x, y):\n    return x + y\n",
    )
    result = verify_candidate(
        task,
        candidate.forward,
        seeds=[0, 1],
        shapes=[(16,)],
        device="cpu",
        dtype="float32",
    )
    assert result.passed
    assert all(case.passed for case in result.cases)


def test_verifier_fails_on_wrong_candidate(tmp_path):
    task = get_task("vector_add")
    candidate = _load_tmp_candidate(
        tmp_path,
        "def forward(x, y):\n    return x - y\n",
    )
    result = verify_candidate(
        task,
        candidate.forward,
        seeds=[0],
        shapes=[(16,)],
        device="cpu",
        dtype="float32",
    )
    assert not result.passed
    assert result.cases[0].error_type == "values_not_close"
