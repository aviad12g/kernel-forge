"""Static candidate policy checks for anti-cheating guardrails."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass
class CandidatePolicyResult:
    """Structured result from lightweight static candidate checks."""

    passed: bool
    warnings: list[str] = field(default_factory=list)
    rejection_reason: str | None = None
    has_forward: bool = False
    uses_triton: bool = False


def check_candidate_policy(code: str, *, allow_torch_fallback: bool) -> CandidatePolicyResult:
    """Reject obvious reference/fallback candidates before verification.

    This is intentionally conservative. It catches clear PyTorch fallback and
    reference-calling patterns when fallback mode is disabled, while warning on
    uncertain cases instead of blocking them.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return CandidatePolicyResult(
            passed=False,
            rejection_reason=f"syntax_error at line {exc.lineno}: {exc.msg}",
        )

    forward = _find_forward(tree)
    uses_triton = _uses_triton(tree)
    warnings: list[str] = []

    if forward is None:
        return CandidatePolicyResult(
            passed=False,
            rejection_reason="missing_forward",
            has_forward=False,
            uses_triton=uses_triton,
        )

    suspicious_import = _find_suspicious_import(tree)
    suspicious_call = _find_suspicious_call(forward)
    direct_fallback = _find_direct_torch_fallback(forward)

    if suspicious_import:
        return CandidatePolicyResult(
            passed=False,
            rejection_reason=suspicious_import,
            warnings=warnings,
            has_forward=True,
            uses_triton=uses_triton,
        )

    if suspicious_call:
        return CandidatePolicyResult(
            passed=False,
            rejection_reason=suspicious_call,
            warnings=warnings,
            has_forward=True,
            uses_triton=uses_triton,
        )

    if not allow_torch_fallback and direct_fallback:
        return CandidatePolicyResult(
            passed=False,
            rejection_reason=direct_fallback,
            warnings=warnings,
            has_forward=True,
            uses_triton=uses_triton,
        )

    if not allow_torch_fallback and not uses_triton:
        warnings.append("no_triton_kernel_launch_detected")
    if allow_torch_fallback and direct_fallback:
        warnings.append("torch_fallback_allowed")

    return CandidatePolicyResult(
        passed=True,
        warnings=warnings,
        has_forward=True,
        uses_triton=uses_triton,
    )


def _find_forward(tree: ast.Module) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            return node
    return None


def _uses_triton(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            if any(name.startswith("triton") for name in names):
                return True
        if isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            module = node.module or ""
            if module.startswith("triton") or any(name.startswith("triton") for name in names):
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            return True
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if _name_of(decorator) in {"triton.jit", "jit", "tl.jit"}:
                    return True
    return False


def _find_suspicious_import(tree: ast.Module) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "openkernelforge.tasks" in module or "simple_tasks" in module:
                return "imports_reference_or_task_module"
            for alias in node.names:
                lowered = alias.name.lower()
                if "reference" in lowered or lowered.startswith("ref_"):
                    return "imports_reference_or_task_module"
        if isinstance(node, ast.Import):
            for alias in node.names:
                lowered = alias.name.lower()
                if "openkernelforge.tasks" in lowered or "simple_tasks" in lowered:
                    return "imports_reference_or_task_module"
    return None


def _find_suspicious_call(forward: ast.FunctionDef) -> str | None:
    for node in ast.walk(forward):
        if not isinstance(node, ast.Call):
            continue
        call_name = _name_of(node.func).lower()
        leaf = call_name.split(".")[-1]
        if leaf in {"reference", "ref_forward", "torch_forward"}:
            return f"calls_suspicious_function:{call_name}"
        if "reference" in leaf or leaf.startswith("ref_"):
            return f"calls_suspicious_function:{call_name}"
    return None


def _find_direct_torch_fallback(forward: ast.FunctionDef) -> str | None:
    for node in ast.walk(forward):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        reason = _fallback_expr_reason(node.value)
        if reason:
            return reason
    return None


def _fallback_expr_reason(node: ast.AST) -> str | None:
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            return "obvious_torch_fallback:direct_add"
        if isinstance(node.op, ast.Mult):
            return "obvious_torch_fallback:direct_mul"
        if isinstance(node.op, ast.MatMult):
            return "obvious_torch_fallback:direct_matmul"
    if isinstance(node, ast.Call):
        call_name = _name_of(node.func)
        leaf = call_name.split(".")[-1]
        if call_name in {
            "torch.add",
            "torch.clamp",
            "torch.relu",
            "torch.sigmoid",
            "torch.matmul",
            "torch.mm",
            "torch.mul",
            "torch.multiply",
            "torch.maximum",
            "torch.minimum",
            "torch.sum",
            "torch.nn.functional.relu",
            "F.relu",
            "F.layer_norm",
        } or leaf in {"clamp", "relu", "sigmoid"}:
            return f"obvious_torch_fallback:{call_name}"
        for arg in node.args:
            nested = _fallback_expr_reason(arg)
            if nested:
                return nested
    return None


def _name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_of(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    return ""
