from pathlib import Path

from scripts.package_workshop2026_release import find_secret_like_token


def test_release_secret_scan_accepts_placeholder(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text("export GEMINI_API_KEY=<key>\n", encoding="utf-8")

    assert find_secret_like_token(path) is None


def test_release_secret_scan_rejects_key_like_value(tmp_path: Path) -> None:
    path = tmp_path / "secret.txt"
    path.write_text("GEMINI_API_KEY=not-a-placeholder-secret\n", encoding="utf-8")

    assert find_secret_like_token(path) == "GEMINI_API_KEY"
