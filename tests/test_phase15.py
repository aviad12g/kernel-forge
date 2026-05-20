import io
import json
import tarfile
from pathlib import Path

import pytest

import scripts.import_runpod_artifacts as import_artifacts
import scripts.package_runpod_artifacts as package_artifacts
import scripts.update_artifact_index as update_index
import scripts.validate_research_package as validate_package


def test_package_runpod_artifacts_handles_missing_paths_gracefully(tmp_path):
    source_root = tmp_path / "source"
    run_dir = source_root / "runs/20260519_213349"
    run_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text('{"record_type":"candidate"}\n', encoding="utf-8")
    (run_dir / "environment_probe.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text("agent:\n  type: template\n", encoding="utf-8")
    candidate_dir = run_dir / "candidates/bias_gelu"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "candidate_000.py").write_text("def forward(*args):\n    return args[0]\n", encoding="utf-8")

    archive = package_artifacts.package_artifacts(source_root, tmp_path / "artifacts.tar.gz")
    assert archive.exists()
    with tarfile.open(archive, "r:gz") as tar:
        manifest = json.loads(tar.extractfile("artifact_manifest.json").read().decode("utf-8"))
        names = tar.getnames()
    assert manifest["missing"]
    assert "runs/20260519_213349_template_fused8_wide/results.jsonl" in names


def test_import_runpod_artifacts_rejects_unsafe_archive_paths(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = b"bad"
        info = tarfile.TarInfo("../evil.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError):
        import_artifacts.import_artifacts(archive, tmp_path / "out")


def test_validate_research_package_catches_fake_secret_strings(tmp_path):
    _write_required_reports(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    fake_secret = "OPENAI_" + "API_KEY=" + "sk-" + "testsecretvalue123456\n"
    (artifacts / "leak.txt").write_text(fake_secret, encoding="utf-8")
    ok, errors, warnings, report = validate_package.validate_research_package(tmp_path)
    assert not ok
    assert any("possible secret" in error for error in errors)
    assert report.exists()


def test_validate_research_package_passes_minimal_synthetic_package(tmp_path):
    _write_required_reports(tmp_path)
    run_dir = tmp_path / "artifacts/runs/20260519_213349_template_fused8_wide"
    run_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text('{"record_type":"candidate"}\n', encoding="utf-8")
    (run_dir / "environment_probe.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "analysis.md").write_text("# Analysis\n", encoding="utf-8")
    dataset = tmp_path / "artifacts/datasets/fused8_curated_v1"
    dataset.mkdir(parents=True)
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    (dataset / "correct_fast_repeat_stable.jsonl").write_text("", encoding="utf-8")
    ok, errors, warnings, report = validate_package.validate_research_package(tmp_path)
    assert ok
    assert errors == []
    assert warnings
    assert "PASS" in report.read_text(encoding="utf-8")


def test_update_artifact_index_marks_missing_artifacts(tmp_path):
    (tmp_path / "reports").mkdir()
    path = update_index.update_artifact_index(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "missing" in text
    assert "optional missing" in text


def test_release_checklist_exists():
    assert Path("reports/release_checklist.md").exists()


def _write_required_reports(root: Path) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True)
    for rel in validate_package.REQUIRED_REPORTS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Report\nNo SOTA claim.\n", encoding="utf-8")
