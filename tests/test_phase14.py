import csv
from pathlib import Path

from openkernelforge.reports.phase14 import (
    TABLE_NAMES,
    build_phase14_report,
    check_research_artifacts,
)
import scripts.build_phase14_report as build_script
import scripts.check_research_artifacts as check_script


def test_build_phase14_report_creates_expected_files(tmp_path):
    (tmp_path / "README.md").write_text("OpenKernelForge\n\nNo SOTA claim.\n", encoding="utf-8")
    written = build_phase14_report(tmp_path)
    written_names = {path.name for path in written}
    assert "openkernelforge_technical_report.md" in written_names
    assert "reproducibility.md" in written_names
    assert "artifact_index.md" in written_names
    for name in TABLE_NAMES:
        assert (tmp_path / "reports" / "tables" / name).exists()
    report = (tmp_path / "reports/openkernelforge_technical_report.md").read_text(encoding="utf-8")
    assert "internal fused8" in report
    assert "not a SOTA claim" in report


def test_phase14_csv_files_are_valid(tmp_path):
    (tmp_path / "README.md").write_text("No SOTA claim.\n", encoding="utf-8")
    build_phase14_report(tmp_path)
    for name in TABLE_NAMES:
        path = tmp_path / "reports" / "tables" / name
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows, name


def test_artifact_checker_works(tmp_path):
    (tmp_path / "README.md").write_text("No SOTA claim.\n", encoding="utf-8")
    build_phase14_report(tmp_path)
    ok, errors, warnings = check_research_artifacts(tmp_path)
    assert ok
    assert errors == []
    assert "datasets/fused8_curated_v1 is not present" in "\n".join(warnings)


def test_phase14_scripts_return_success(tmp_path, capsys):
    (tmp_path / "README.md").write_text("No SOTA claim.\n", encoding="utf-8")
    assert build_script.build_report_main(["--root", str(tmp_path)]) == 0
    assert check_script.check_artifacts_main(["--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Generated research artifacts" in out
    assert "Artifact check passed" in out


def test_readme_contains_no_sota_limitation():
    text = Path("README.md").read_text(encoding="utf-8").lower()
    assert "not a sota claim" in text or "no sota claim" in text or "no sota claims" in text
