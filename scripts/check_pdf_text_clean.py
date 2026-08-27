from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_TARGETS = (
    (
        "long paper",
        ROOT / "paper" / "openkernelforge_paper.pdf",
        ROOT / "paper" / "overleaf",
    ),
    (
        "workshop paper",
        ROOT / "paper" / "workshop2026" / "openkernelforge_workshop2026.pdf",
        ROOT / "paper" / "workshop2026",
    ),
)


BAD_CHARS = {
    "\u00ad": "soft hyphen U+00AD",
    "\ufffd": "replacement character U+FFFD",
    "\ufffc": "object replacement character U+FFFC",
    "\ufffe": "noncharacter U+FFFE",
}


def main() -> int:
    errors: list[str] = []
    checked: list[str] = []
    for name, pdf, source_root in PAPER_TARGETS:
        text, source = _extract_pdf_text(pdf)
        if text is None:
            text = _source_text(source_root)
            source = "TeX source fallback"
        checked.append(f"{name} via {source}")
        errors.extend(_validate_text(text, name))

    if errors:
        print("PDF text cleanliness check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("PDF text cleanliness check passed: " + "; ".join(checked) + ".")
    return 0


def _validate_text(text: str, document_name: str) -> list[str]:
    errors: list[str] = []
    for char, char_name in BAD_CHARS.items():
        if char in text:
            errors.append(f"{document_name}: {char_name} found")
    control_codes = sorted(
        {ord(char) for char in text if ord(char) < 32 and char not in "\t\n\r\f"}
    )
    if control_codes:
        rendered = ", ".join(f"U+{code:04X}" for code in control_codes)
        errors.append(f"{document_name}: unexpected control characters found: {rendered}")

    lowered = text.lower()
    forbidden_plain = [
        "todo citation",
        "citation todo",
        "kernelbench pending",
        "achieves sota",
        "sota performance",
    ]
    for phrase in forbidden_plain:
        if phrase in lowered:
            errors.append(f"{document_name}: forbidden phrase found: {phrase}")

    for phrase in ["state-of-the-art", "state of the art", "full kernelbench"]:
        for match in re.finditer(re.escape(phrase), lowered):
            context = lowered[max(0, match.start() - 80): match.end() + 80]
            if not _is_negated(context):
                errors.append(f"{document_name}: unsupported unnegated phrase found: {phrase}")
                break

    references_pos = _first_section_pos(lowered, "references")
    appendix_pos = _appendix_section_pos(lowered)
    if references_pos != -1 and appendix_pos != -1 and appendix_pos < references_pos:
        errors.append(f"{document_name}: appendix appears before references in extracted text")

    if references_pos != -1 and appendix_pos != -1:
        references_block = lowered[references_pos:appendix_pos]
        if "appendix table" in references_block or "example generated candidates" in references_block:
            errors.append(f"{document_name}: appendix text appears inside the references block")
    return errors


def _extract_pdf_text(pdf: Path) -> tuple[str | None, str]:
    if not pdf.exists():
        return None, "missing PDF"

    try:
        result = subprocess.run(
            ["pdftotext", str(pdf), "-"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout, "pdftotext"
    except FileNotFoundError:
        pass

    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(pdf))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
        if text.strip():
            return text, "pypdf"
    except Exception:
        pass

    return None, "no PDF text extractor"


def _source_text(source_root: Path) -> str:
    parts: list[str] = []
    if source_root.exists():
        for path in sorted(source_root.rglob("*.tex")):
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    if (ROOT / "paper" / "paper.md").exists():
        parts.append((ROOT / "paper" / "paper.md").read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _is_negated(context: str) -> bool:
    negator_patterns = [
        r"\bno\s+",
        r"\bnot\s+",
        r"\bdo\s+not\s+",
        r"\bdoes\s+not\s+",
        r"\bwe\s+do\s+not\s+claim\b",
        r"\bnot\s+designed\s+to\s+estimate\b",
        r"\bwithout\s+claiming\b",
        r"\bneither\s+(?:a|an)\s+",
    ]
    return any(re.search(pattern, context) for pattern in negator_patterns)


def _first_section_pos(text: str, section: str) -> int:
    candidates = [
        f"\n{section}\n",
        f"\n{section} ",
        f"\n{section}\f",
    ]
    positions = [text.find(candidate) for candidate in candidates if text.find(candidate) != -1]
    return min(positions) if positions else -1


def _appendix_section_pos(text: str) -> int:
    """Locate the rendered appendix heading, not prose references to appendices."""

    heading_patterns = [
        r"\n(?:[a-z]\s+)?supplementary material\n",
        r"\n(?:[a-z]\s+)?appendices\n",
        r"\n[a-z]\s+preregistered artifact contract\n",
    ]
    positions: list[int] = []
    for pattern in heading_patterns:
        match = re.search(pattern, text)
        if match:
            positions.append(match.start())
    return min(positions) if positions else -1


if __name__ == "__main__":
    raise SystemExit(main())
