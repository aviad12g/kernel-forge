import io
import hashlib
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


def test_import_runpod_artifacts_rejects_symlinks(tmp_path):
    archive = tmp_path / "symlink.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("linked-secret")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    with pytest.raises(ValueError, match="unsupported link or special file"):
        import_artifacts.import_artifacts(archive, tmp_path / "out")


def test_import_runpod_artifacts_verifies_packaged_checksums(tmp_path):
    archive = tmp_path / "bad-checksum.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = b"actual"
        file_info = tarfile.TarInfo("artifact.txt")
        file_info.size = len(payload)
        tar.addfile(file_info, io.BytesIO(payload))
        checksum = b"0" * 64 + b"  artifact.txt\n"
        checksum_info = tarfile.TarInfo("SHA256SUMS")
        checksum_info.size = len(checksum)
        tar.addfile(checksum_info, io.BytesIO(checksum))
    with pytest.raises(ValueError, match="checksum mismatch"):
        import_artifacts.import_artifacts(archive, tmp_path / "out")


def test_import_runpod_artifacts_requires_checksums(tmp_path):
    archive = tmp_path / "no-checksum.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = b"artifact"
        info = tarfile.TarInfo("artifact.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="missing SHA256SUMS"):
        import_artifacts.import_artifacts(archive, tmp_path / "out")


def test_import_runpod_artifacts_rejects_unchecksummed_files(tmp_path):
    archive = tmp_path / "extra-file.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        listed = b"listed"
        listed_info = tarfile.TarInfo("listed.txt")
        listed_info.size = len(listed)
        tar.addfile(listed_info, io.BytesIO(listed))
        extra = b"extra"
        extra_info = tarfile.TarInfo("extra.txt")
        extra_info.size = len(extra)
        tar.addfile(extra_info, io.BytesIO(extra))
        checksum = f"{hashlib.sha256(listed).hexdigest()}  listed.txt\n".encode()
        checksum_info = tarfile.TarInfo("SHA256SUMS")
        checksum_info.size = len(checksum)
        tar.addfile(checksum_info, io.BytesIO(checksum))

    with pytest.raises(ValueError, match="unchecksummed files"):
        import_artifacts.import_artifacts(archive, tmp_path / "out")


def test_archive_import_replaces_stale_destination_and_writes_valid_checksums(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    archive = package_artifacts.package_artifacts(source_root, tmp_path / "artifacts.tar.gz")
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.txt").write_text("stale", encoding="utf-8")

    import_artifacts.import_artifacts(archive, out)

    assert not (out / "stale.txt").exists()
    assert import_artifacts._verify_packaged_checksums(out) > 0


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


def test_validate_research_package_strict_mode_requires_all_raw_runs(tmp_path):
    _write_required_reports(tmp_path)
    (tmp_path / "artifacts").mkdir()
    ok, errors, _, _ = validate_package.validate_research_package(tmp_path, strict=True)
    assert not ok
    assert any("missing imported run artifact" in error for error in errors)


def test_kernelbench_run_validation_does_not_require_fused8_report(tmp_path):
    run_dir = tmp_path / "kernelbench_run"
    run_dir.mkdir()
    (run_dir / "results.jsonl").write_text('{"record_type":"candidate"}\n', encoding="utf-8")
    (run_dir / "environment_probe.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "kernelbench_l1_check.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "kernelbench_l1_check.md").write_text("# KernelBench check\n", encoding="utf-8")
    errors: list[str] = []
    warnings: list[str] = []

    validate_package._validate_run_dir(run_dir, errors, warnings)

    assert errors == []
    assert warnings == []


def test_update_artifact_index_marks_missing_artifacts(tmp_path):
    (tmp_path / "reports").mkdir()
    path = update_index.update_artifact_index(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "missing" in text
    assert "optional missing" in text


def test_update_artifact_index_uses_real_workspace_path(tmp_path):
    (tmp_path / "reports").mkdir()
    config = tmp_path / "configs/kernelbench_l1_20task_rigorous_safe.yaml"
    config.parent.mkdir()
    config.write_text("tasks: []\n", encoding="utf-8")

    path = update_index.update_artifact_index(tmp_path)
    text = path.read_text(encoding="utf-8")

    assert f"`{config}` | present in workspace" in text
    assert (
        f"`{tmp_path / 'artifacts/runpod_imports/configs/kernelbench_l1_20task_rigorous_safe.yaml'}` "
        "| present in workspace"
    ) not in text


def test_release_checklist_exists():
    assert Path("reports/release_checklist.md").exists()


def _write_required_reports(root: Path) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True)
    for rel in validate_package.REQUIRED_REPORTS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Report\nNo SOTA claim.\n", encoding="utf-8")
