from pathlib import Path

from scripts.check_kernelbench_claim_status import check_claim_status


def test_kernelbench_claim_status_passes_for_repo():
    ok, errors = check_claim_status(Path.cwd())
    assert ok, "\n".join(errors)


def test_kernelbench_claim_status_reports_missing_files(tmp_path):
    ok, errors = check_claim_status(tmp_path)
    assert not ok
    assert any("missing public claim file" in error for error in errors)
