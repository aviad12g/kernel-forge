import pytest

from openkernelforge.harness.sandbox import write_candidate_source


def test_write_candidate_source_stays_under_run_directory(tmp_path):
    path = write_candidate_source(tmp_path, "family/task", 0, "def forward(x):\n    return x\n")

    assert path == tmp_path / "candidates" / "family" / "task" / "candidate_000.py"
    assert path.exists()


@pytest.mark.parametrize("task_id", ["../../escape", "/tmp/escape"])
def test_write_candidate_source_rejects_path_escape(tmp_path, task_id):
    with pytest.raises(ValueError, match="Unsafe task id"):
        write_candidate_source(tmp_path, task_id, 0, "def forward(x):\n    return x\n")


def test_write_candidate_source_rejects_negative_index(tmp_path):
    with pytest.raises(ValueError, match="candidate_index must be non-negative"):
        write_candidate_source(tmp_path, "task", -1, "def forward(x):\n    return x\n")
