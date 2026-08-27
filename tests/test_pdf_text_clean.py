from __future__ import annotations

from scripts.check_pdf_text_clean import _is_negated, _validate_text


def test_pdf_negation_allows_line_wrapping() -> None:
    assert _is_negated("we make no\nfull-kernelbench or state-of-the-art claim")
    assert _is_negated("this does\nnot establish full kernelbench performance")
    assert not _is_negated("we establish state-of-the-art performance")


def test_pdf_text_rejects_unexpected_control_characters() -> None:
    assert _validate_text("ordinary text\fnext page", "paper") == []
    assert _validate_text("bad \x10 glyph", "paper") == [
        "paper: unexpected control characters found: U+0010"
    ]
