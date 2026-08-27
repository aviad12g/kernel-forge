from __future__ import annotations

import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCAN_GLOBS = [
    "paper/overleaf/**/*.tex",
    "paper/workshop2026/**/*.tex",
    "paper/*.tex",
    "paper/*.md",
    "scripts/build_paper_pdf.py",
    "scripts/build_paper_assets.py",
    "scripts/make_paper_figures.py",
    "scripts/build_workshop2026_paper.py",
]

DISALLOWED = {
    "\u00ad": "soft hyphen",
    "\ufffd": "replacement character",
    "\ufffc": "object replacement character",
    "\ufff9": "interlinear annotation anchor",
    "\ufffa": "interlinear annotation separator",
    "\ufffb": "interlinear annotation terminator",
}

ALLOWED_NON_ASCII = {"×"}


def _iter_paths() -> list[Path]:
    paths: set[Path] = set()
    for pattern in SCAN_GLOBS:
        paths.update(ROOT.glob(pattern))
    return sorted(path for path in paths if path.is_file())


def _line_col(text: str, index: int) -> tuple[int, int]:
    line = text.count("\n", 0, index) + 1
    last_newline = text.rfind("\n", 0, index)
    col = index + 1 if last_newline == -1 else index - last_newline
    return line, col


def _describe(ch: str) -> str:
    codepoint = f"U+{ord(ch):04X}"
    name = unicodedata.name(ch, "UNKNOWN")
    reason = DISALLOWED.get(ch)
    if reason:
        return f"{codepoint} {name} ({reason})"
    return f"{codepoint} {name}"


def main() -> int:
    findings: list[str] = []
    for path in _iter_paths():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            findings.append(f"{path.relative_to(ROOT)}: decode error: {exc}")
            continue

        for index, ch in enumerate(text):
            code = ord(ch)
            if ch in DISALLOWED:
                line, col = _line_col(text, index)
                findings.append(f"{path.relative_to(ROOT)}:{line}:{col}: {_describe(ch)}")
                continue
            if ch in {"\n", "\r", "\t"}:
                continue
            if unicodedata.category(ch).startswith("C"):
                line, col = _line_col(text, index)
                findings.append(f"{path.relative_to(ROOT)}:{line}:{col}: {_describe(ch)}")
                continue
            if code > 127 and ch not in ALLOWED_NON_ASCII:
                line, col = _line_col(text, index)
                findings.append(f"{path.relative_to(ROOT)}:{line}:{col}: {_describe(ch)}")

    if findings:
        print("Paper text cleanliness check failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print("Paper text cleanliness check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
