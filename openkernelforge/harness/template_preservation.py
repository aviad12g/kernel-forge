"""Static template-copy preservation checks.

These checks are deliberately heuristic. They are intended to catch obvious
wrapper/fallback regressions when an LLM is asked to copy a known-good Triton
template, not to prove semantic equivalence.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any


FORBIDDEN_TORCH_OPS = {
    "add",
    "matmul",
    "maximum",
    "mul",
    "relu",
    "sigmoid",
}


@dataclass(frozen=True)
class TemplatePreservationResult:
    """Result of a heuristic template preservation check."""

    passed: bool
    score: int
    warnings: list[str] = field(default_factory=list)
    rejection_reason: str | None = None
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "warnings": list(self.warnings),
            "rejection_reason": self.rejection_reason,
            "checks": dict(self.checks),
        }


def check_template_preservation(
    candidate_code: str,
    template_code: str,
    *,
    task_id: str | None = None,
    reject_if_score_below: int = 70,
    reject_fallbacks: bool = True,
    reject_forbidden_torch_ops: bool = True,
) -> TemplatePreservationResult:
    """Compare candidate and source template with transparent static heuristics."""

    candidate = candidate_code or ""
    template = template_code or ""
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    candidate_tree = _parse(candidate, warnings, "candidate")
    template_tree = _parse(template, warnings, "template")
    candidate_forward = _function_node(candidate_tree, "forward") if candidate_tree else None
    template_forward = _function_node(template_tree, "forward") if template_tree else None

    candidate_kernel_launches = _kernel_launch_count(candidate)
    template_kernel_launches = _kernel_launch_count(template)
    candidate_forbidden = sorted(_forbidden_torch_ops(candidate_forward))
    candidate_try_except = _has_try_except(candidate_tree)
    fallback_detected = _fallback_detected(candidate, candidate_forward)
    wrapper_lines = _function_line_count(candidate_forward)
    template_wrapper_lines = _function_line_count(template_forward)

    checks.update(
        {
            "candidate_has_forward": candidate_forward is not None,
            "template_has_forward": template_forward is not None,
            "candidate_has_triton_jit": "@triton.jit" in candidate,
            "template_has_triton_jit": "@triton.jit" in template,
            "candidate_has_block_size": "BLOCK_SIZE" in candidate,
            "template_has_block_size": "BLOCK_SIZE" in template,
            "candidate_kernel_launches": candidate_kernel_launches,
            "template_kernel_launches": template_kernel_launches,
            "forbidden_torch_ops": candidate_forbidden,
            "try_except_present": candidate_try_except,
            "fallback_detected": fallback_detected,
            "wrapper_line_count": wrapper_lines,
            "template_wrapper_line_count": template_wrapper_lines,
            "bias_modulo_indexing": _has_bias_modulo(candidate),
        }
    )

    score = 100
    if candidate_forward is None:
        warnings.append("missing forward function")
        score -= 30
    if "@triton.jit" not in candidate:
        warnings.append("missing @triton.jit")
        score -= 20
    if "BLOCK_SIZE" not in candidate:
        warnings.append("missing BLOCK_SIZE")
        score -= 15
    if template_kernel_launches and candidate_kernel_launches != template_kernel_launches:
        warnings.append(
            f"kernel launch count changed from {template_kernel_launches} to {candidate_kernel_launches}"
        )
        score -= 15
    if candidate_try_except:
        warnings.append("added try/except wrapper logic")
        score -= 15
    if fallback_detected:
        warnings.append("fallback branch or reference-style fallback detected")
        score -= 25
    if candidate_forbidden:
        warnings.append("forbidden torch ops in forward: " + ", ".join(candidate_forbidden))
        score -= 25
    if task_id == "bias_relu" and not _has_bias_modulo(candidate):
        warnings.append("bias_relu candidate does not preserve modulo feature indexing")
        score -= 15
    if template_wrapper_lines and wrapper_lines > template_wrapper_lines + 8:
        warnings.append(
            f"forward wrapper grew from {template_wrapper_lines} to {wrapper_lines} source lines"
        )
        score -= 10

    score = max(0, min(100, score))
    rejection_reason = None
    if reject_forbidden_torch_ops and candidate_forbidden:
        rejection_reason = "template_preservation_forbidden_torch_ops"
    elif reject_fallbacks and fallback_detected:
        rejection_reason = "template_preservation_fallback_detected"
    elif score < reject_if_score_below:
        rejection_reason = f"template_preservation_score_below_{reject_if_score_below}"

    return TemplatePreservationResult(
        passed=rejection_reason is None,
        score=score,
        warnings=warnings,
        rejection_reason=rejection_reason,
        checks=checks,
    )


def _parse(source: str, warnings: list[str], label: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        warnings.append(f"{label} syntax error while checking preservation: {exc}")
        return None


def _function_node(tree: ast.Module | None, name: str) -> ast.FunctionDef | None:
    if tree is None:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _forbidden_torch_ops(forward_node: ast.FunctionDef | None) -> set[str]:
    if forward_node is None:
        return set()
    ops: set[str] = set()
    for node in ast.walk(forward_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
            and func.attr in FORBIDDEN_TORCH_OPS
        ):
            ops.add(f"torch.{func.attr}")
    return ops


def _has_try_except(tree: ast.Module | None) -> bool:
    return bool(tree and any(isinstance(node, ast.Try) for node in ast.walk(tree)))


def _fallback_detected(source: str, forward_node: ast.FunctionDef | None) -> bool:
    lowered = source.lower()
    if any(token in lowered for token in ("fallback", "reference", "ref_forward", "torch_forward")):
        return True
    if "except importerror" in lowered or "except exception" in lowered:
        return True
    return bool(_forbidden_torch_ops(forward_node))


def _kernel_launch_count(source: str) -> int:
    return len(re.findall(r"\w+\s*\[\s*grid\s*\]\s*\(", source))


def _function_line_count(node: ast.FunctionDef | None) -> int:
    if node is None or not hasattr(node, "end_lineno"):
        return 0
    return int(node.end_lineno or node.lineno) - int(node.lineno) + 1


def _has_bias_modulo(source: str) -> bool:
    compact = re.sub(r"\s+", "", source)
    return "%feature_dim" in compact or "%features" in compact or "%bias.numel()" in compact
