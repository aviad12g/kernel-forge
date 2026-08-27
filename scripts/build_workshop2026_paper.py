#!/usr/bin/env python3
"""Build and enforce invariants for the four-page workshop paper."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "workshop2026"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-ready", action="store_true")
    parser.add_argument(
        "--submission-upload",
        action="store_true",
        help="build the venue-upload PDF with the official style notice",
    )
    args = parser.parse_args()
    if not shutil.which("latexmk"):
        raise RuntimeError("latexmk is required for the formal workshop build")
    if args.submission_ready or args.submission_upload:
        _reject_pending_markers()
    source = "submission.tex" if args.submission_upload else "main.tex"
    stem = Path(source).stem
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", source],
        cwd=PAPER,
        check=True,
    )
    main_pages = _main_matter_page(PAPER / f"{stem}.aux")
    if main_pages > 4:
        raise RuntimeError(f"workshop main matter is {main_pages} pages; strict limit is 4")
    if args.submission_upload:
        output = PAPER / "openkernelforge_workshop2026_submission.pdf"
        shutil.copy2(PAPER / f"{stem}.pdf", output)
        print(f"submission-upload PDF: {output}")
        print(f"main matter pages: {main_pages}/4")
        print("submission status: upload artifact prepared; no venue submission performed")
        return 0
    output = PAPER / "workshop2026_draft.pdf"
    review_output = PAPER / "openkernelforge_workshop2026.pdf"
    shutil.copy2(PAPER / f"{stem}.pdf", output)
    shutil.copy2(PAPER / f"{stem}.pdf", review_output)
    print(f"workshop PDF: {review_output}")
    print(f"compatibility copy: {output}")
    print(f"main matter pages: {main_pages}/4")
    print(f"submission ready: {args.submission_ready}")
    return 0


def _reject_pending_markers() -> None:
    markers = ("pendingcampaign", "Pending corrected campaign", "PENDING_CORRECTED_CAMPAIGN")
    findings: list[str] = []
    for path in sorted(PAPER.rglob("*.tex")):
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker in text:
                findings.append(f"{path.relative_to(ROOT)}: {marker}")
    if findings:
        raise RuntimeError("submission-ready build contains pending evidence markers:\n" + "\n".join(findings))


def _main_matter_page(aux_path: Path) -> int:
    aux = aux_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"\\newlabel\{mainmatterend\}\{\{[^}]*\}\{(\d+)\}",
        aux,
    )
    if not match:
        raise RuntimeError("main-matter page label was not emitted")
    return int(match.group(1))


if __name__ == "__main__":
    raise SystemExit(main())
