"""Utilities for extracting candidate Python code from model responses."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CodeExtractionResult:
    """Result of turning a raw model response into candidate source code."""

    code: str | None
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.code is not None and self.error is None


_FENCE_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+.-]*)\s*\n(?P<code>.*?)```", re.DOTALL)
_CODE_START_RE = re.compile(r"^\s*(import\s+|from\s+|def\s+|class\s+|@)")


def extract_python_code(response: str) -> CodeExtractionResult:
    """Extract a valid Python candidate exposing ``forward`` or ``ModelNew``.

    The extractor accepts raw Python, fenced Markdown blocks, and responses with
    short explanation before or after the code. It returns an error instead of
    guessing when no syntactically valid ``forward`` implementation is found.
    """

    fences = list(_FENCE_RE.finditer(response))
    candidates: list[tuple[str, dict[str, object]]] = []

    for index, match in enumerate(fences):
        language = match.group("lang").strip().lower()
        code = match.group("code").strip()
        candidates.append(
            (
                code,
                {
                    "source": "fenced",
                    "language": language or "plain",
                    "fence_count": len(fences),
                    "selected_index": index,
                },
            )
        )

    stripped = response.strip()
    if stripped:
        candidates.append(
            (
                stripped,
                {"source": "raw", "fence_count": len(fences), "selected_index": None},
            )
        )
        trimmed = _trim_response_to_code(stripped)
        if trimmed and trimmed != stripped:
            candidates.append(
                (
                    trimmed,
                    {
                        "source": "raw_trimmed",
                        "fence_count": len(fences),
                        "selected_index": None,
                    },
                )
            )

    parse_errors: list[str] = []
    for code, metadata in candidates:
        parsed = _parse_candidate(code)
        if parsed is not None:
            parse_errors.append(parsed)
            continue
        if not _has_forward_function(code):
            parse_errors.append("candidate does not define forward")
            continue
        return CodeExtractionResult(
            code=code.strip() + "\n",
            metadata={**metadata, "has_forward": True},
        )

    details = "; ".join(parse_errors[-3:]) if parse_errors else "response was empty"
    return CodeExtractionResult(
        code=None,
        error=f"No usable Python candidate exposing forward was found: {details}",
        metadata={"fence_count": len(fences), "has_forward": False},
    )


def _trim_response_to_code(response: str) -> str | None:
    lines = response.splitlines()
    start = None
    for index, line in enumerate(lines):
        if _CODE_START_RE.match(line):
            start = index
            break
    if start is None:
        return None

    code_lines = lines[start:]
    while code_lines:
        candidate = "\n".join(code_lines).strip()
        if _parse_candidate(candidate) is None and _has_forward_function(candidate):
            return candidate
        code_lines = code_lines[:-1]
    return None


def _parse_candidate(code: str) -> str | None:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error at line {exc.lineno}: {exc.msg}"
    return None


def _has_forward_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward":
            return True
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            if any(
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == "forward"
                for child in node.body
            ):
                return True
    return False
