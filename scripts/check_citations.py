from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = (
    ROOT / "paper" / "overleaf",
    ROOT / "paper" / "workshop2026",
)


def main() -> int:
    errors: list[str] = []
    summaries: list[str] = []
    unused_messages: list[str] = []
    for tex_root in PROJECTS:
        tex_keys = _cite_keys(tex_root)
        bib_keys = _bib_keys(tex_root / "references.bib")
        missing = sorted(tex_keys - bib_keys)
        unused = sorted(bib_keys - tex_keys)
        label = tex_root.relative_to(ROOT).as_posix()
        if missing:
            errors.append(f"{label} missing citation keys: " + ", ".join(missing))
        todo_hits = _todo_citation_hits(tex_root)
        if todo_hits:
            errors.append(f"{label} TODO citation text: " + ", ".join(todo_hits))
        summaries.append(f"{label}: {len(tex_keys)} cited keys, {len(bib_keys)} entries")
        if unused:
            unused_messages.append(f"{label} unused: " + ", ".join(unused))
    if errors:
        for error in errors:
            print(error)
        return 1
    print("Citation check passed: " + "; ".join(summaries) + ".")
    for message in unused_messages:
        print(message)
    return 0


def _cite_keys(tex_root: Path) -> set[str]:
    keys: set[str] = set()
    pattern = re.compile(r"\\cite[a-zA-Z*]*\{([^}]+)\}")
    for path in sorted(tex_root.rglob("*.tex")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def _bib_keys(bib: Path) -> set[str]:
    if not bib.exists():
        return set()
    text = bib.read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", text))


def _todo_citation_hits(tex_root: Path) -> list[str]:
    hits = []
    for path in sorted(tex_root.rglob("*.tex")):
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        if "todo citation" in text or "citation todo" in text:
            hits.append(str(path.relative_to(ROOT)))
    return hits


if __name__ == "__main__":
    raise SystemExit(main())
