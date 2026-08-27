from pathlib import Path

import scripts.check_methodology_docs as check_methodology_docs


def test_methodology_docs_check_passes_for_repo():
    ok, errors = check_methodology_docs.check_methodology_docs(Path.cwd())
    assert ok, "\n".join(errors)


def test_methodology_docs_checker_reports_missing_file(tmp_path):
    ok, errors = check_methodology_docs.check_methodology_docs(tmp_path)
    assert not ok
    assert any("repeatability_label_spec.md" in error for error in errors)
