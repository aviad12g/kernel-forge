"""Static candidate policy checks for anti-cheating guardrails."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


_ALLOWED_IMPORTS = {
    "math",
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "triton",
    "triton.language",
    "typing",
}
_ALLOWED_FROM_TORCH_NAMES = {"Tensor", "nn"}
_ALLOWED_IMPORT_TIME_CALLS = {
    "triton.Config",
    "triton.autotune",
    "triton.heuristics",
    "triton.jit",
}
_ALLOWED_TORCH_CALLS = {
    "torch.empty",
    "torch.empty_like",
    "torch.empty_strided",
    "torch.zeros",
    "torch.zeros_like",
    "torch.ones",
    "torch.ones_like",
    "torch.full",
    "torch.full_like",
    "torch.device",
}
_ALLOWED_TENSOR_METHODS = {
    "data_ptr",
    "dim",
    "element_size",
    "get_device",
    "is_contiguous",
    "is_floating_point",
    "ndimension",
    "new_empty",
    "new_full",
    "new_ones",
    "new_zeros",
    "numel",
    "size",
    "storage_offset",
    "stride",
}
_ALLOWED_TENSOR_ATTRIBUTES = {
    "device",
    "dtype",
    "is_cuda",
    "layout",
    "ndim",
    "requires_grad",
    "shape",
}
_TORCH_COMPUTE_METHODS = {
    "abs",
    "add",
    "argmax",
    "clamp",
    "conv1d",
    "conv2d",
    "conv3d",
    "cross_entropy",
    "div",
    "exp",
    "gelu",
    "kl_div",
    "layer_norm",
    "log",
    "log_softmax",
    "matmul",
    "max",
    "mean",
    "min",
    "mm",
    "mul",
    "multiply",
    "norm",
    "pow",
    "relu",
    "sigmoid",
    "softmax",
    "sqrt",
    "sub",
    "sum",
    "tanh",
    "triplet_margin_loss",
}
_TORCH_COMPUTE_METHODS.update(
    {f"{name}_" for name in tuple(_TORCH_COMPUTE_METHODS)}
    | {
        "copy_",
        "fill_",
        "index_add_",
        "index_copy_",
        "masked_fill_",
        "scatter_",
        "scatter_add_",
        "zero_",
    }
)
_BANNED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "os.popen",
    "os.system",
    "quit",
    "setattr",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "subprocess.run",
    "sys.exit",
    "vars",
}


@dataclass
class CandidatePolicyResult:
    """Structured result from conservative static candidate checks."""

    passed: bool
    warnings: list[str] = field(default_factory=list)
    rejection_reason: str | None = None
    has_forward: bool = False
    uses_triton: bool = False
    policy_version: str = "ast-v5"


def check_candidate_policy(
    code: str,
    *,
    allow_torch_fallback: bool,
    require_triton: bool = False,
) -> CandidatePolicyResult:
    """Reject unsafe imports, reference calls, and high-level compute fallbacks.

    This check is a guardrail, not an operating-system sandbox. Candidates that
    pass still need isolated execution for protection from hangs or invalid GPU
    kernels.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return CandidatePolicyResult(
            passed=False,
            rejection_reason=f"syntax_error at line {exc.lineno}: {exc.msg}",
        )

    forward = _find_forward(tree)
    warnings: list[str] = []

    if forward is None:
        return CandidatePolicyResult(
            passed=False,
            rejection_reason="missing_forward",
            has_forward=False,
            uses_triton=False,
        )

    reachable_functions = _reachable_host_functions(tree, forward)
    uses_triton = _uses_triton(tree, reachable_functions)

    rejection = _find_disallowed_import(tree, strict=not allow_torch_fallback)
    if rejection is None:
        rejection = _find_banned_call(tree)
    if rejection is None:
        rejection = _find_import_time_side_effect(tree)
    if rejection is None:
        rejection = _find_module_compute_alias(tree)
    fallback = None
    if rejection is None:
        for function in reachable_functions:
            rejection = _find_suspicious_call(function)
            if rejection:
                break
            fallback = _find_torch_fallback(function)
            if fallback:
                break

    if rejection:
        return CandidatePolicyResult(
            passed=False,
            rejection_reason=rejection,
            warnings=warnings,
            has_forward=True,
            uses_triton=uses_triton,
        )

    if not allow_torch_fallback and fallback:
        return CandidatePolicyResult(
            passed=False,
            rejection_reason=fallback,
            warnings=warnings,
            has_forward=True,
            uses_triton=uses_triton,
        )

    if require_triton and not uses_triton:
        return CandidatePolicyResult(
            passed=False,
            rejection_reason="missing_triton_kernel_launch",
            warnings=warnings,
            has_forward=True,
            uses_triton=False,
        )
    if not uses_triton:
        warnings.append("no_triton_kernel_launch_detected")
    if allow_torch_fallback and fallback:
        warnings.append("torch_fallback_allowed")

    return CandidatePolicyResult(
        passed=True,
        warnings=warnings,
        has_forward=True,
        uses_triton=uses_triton,
    )


def _find_forward(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    model_new_forward = None
    module_forward = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward":
            module_forward = node
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "forward":
                    model_new_forward = child
                    break
    return model_new_forward or module_forward


def _reachable_host_functions(
    tree: ast.Module,
    forward: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return Python functions reachable from the candidate entry point."""

    top_level = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not _is_triton_kernel(node)
    }
    model_methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            model_methods = {
                child.name: child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            break

    reachable: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    pending: list[ast.FunctionDef | ast.AsyncFunctionDef] = [forward]
    seen: set[int] = set()
    while pending:
        function = pending.pop(0)
        if id(function) in seen:
            continue
        seen.add(id(function))
        reachable.append(function)
        for walk_node in ast.walk(function):
            if not isinstance(walk_node, ast.Call):
                continue
            name = _name_of(walk_node.func)
            helper = top_level.get(name)
            if helper is None and name.startswith("self."):
                helper = model_methods.get(name.split(".")[-1])
            if helper is not None and id(helper) not in seen:
                pending.append(helper)
    return reachable


def _is_triton_kernel(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    decorator_names = {_name_of(decorator) for decorator in node.decorator_list}
    return any(
        name in {"triton.jit", "jit", "tl.jit"} or name.endswith(".jit")
        for name in decorator_names
    )


def _uses_triton(
    tree: ast.Module,
    reachable_functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
) -> bool:
    kernels: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _is_triton_kernel(node):
            kernels.add(node.name)
    if not kernels:
        return False
    for function in reachable_functions:
        for node in ast.walk(function):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Subscript):
                continue
            launched = _name_of(node.func.value).split(".")[-1]
            if launched in kernels:
                return True
    return False


def _find_disallowed_import(tree: ast.Module, *, strict: bool) -> str | None:
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        for module in modules:
            lowered = module.lower()
            if "openkernelforge.tasks" in lowered or "simple_tasks" in lowered or "kernelbench" in lowered:
                return "imports_reference_or_task_module"
            if strict and not _is_allowed_import(lowered):
                return f"disallowed_import:{module}"
        if strict and isinstance(node, ast.Import):
            for alias in node.names:
                expected_alias = {
                    "torch": None,
                    "torch.nn": "nn",
                    "torch.nn.functional": "F",
                    "triton": None,
                    "triton.language": "tl",
                }.get(alias.name)
                if alias.name in {
                    "torch",
                    "torch.nn",
                    "torch.nn.functional",
                    "triton",
                    "triton.language",
                } and alias.asname not in {None, expected_alias}:
                    return f"disallowed_import_alias:{alias.name} as {alias.asname}"
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                lowered = alias.name.lower()
                if "reference" in lowered or lowered.startswith("ref_"):
                    return "imports_reference_or_task_module"
                if strict and (node.module or "") == "torch" and alias.name not in _ALLOWED_FROM_TORCH_NAMES:
                    return f"disallowed_from_torch_import:{alias.name}"
                if strict and (node.module or "").startswith("torch."):
                    return f"disallowed_from_torch_import:{alias.name}"
    return None


def _is_allowed_import(module: str) -> bool:
    return module in _ALLOWED_IMPORTS or module.startswith("triton.")


def _find_banned_call(tree: ast.Module) -> str | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _name_of(node.func)
        if call_name in _BANNED_CALLS:
            return f"unsafe_call:{call_name}"
    return None


def _find_import_time_side_effect(tree: ast.Module) -> str | None:
    """Reject calls executed by module or class definition evaluation."""

    visitor = _ImportTimeCallVisitor()
    visitor.visit(tree)
    return visitor.rejection


class _ImportTimeCallVisitor(ast.NodeVisitor):
    """Visit import-time expressions while deliberately skipping function bodies."""

    def __init__(self) -> None:
        self.rejection: str | None = None

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast visitor API
        if self.rejection is not None:
            return
        call_name = _name_of(node.func)
        if call_name not in _ALLOWED_IMPORT_TIME_CALLS:
            self.rejection = f"import_time_call:{call_name or '<unknown>'}"
            return
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function_definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function_definition(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        for statement in node.body:
            self.visit(statement)

    def _visit_function_definition(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for expression in node.decorator_list:
            self.visit(expression)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            if argument.annotation is not None:
                self.visit(argument.annotation)


def _find_module_compute_alias(tree: ast.Module) -> str | None:
    """Reject module bindings that hide a Torch compute function behind a name."""

    for statement in tree.body:
        value = getattr(statement, "value", None)
        if not isinstance(value, ast.Attribute):
            continue
        name = _name_of(value)
        leaf = name.split(".")[-1]
        if leaf in _TORCH_COMPUTE_METHODS and (
            name.startswith("torch.")
            or name.startswith("F.")
            or name.startswith("nn.functional.")
        ):
            return f"obvious_torch_fallback:bound_alias:{name}"
    return None


def _find_suspicious_call(forward: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    for node in ast.walk(forward):
        if not isinstance(node, ast.Call):
            continue
        call_name = _name_of(node.func)
        leaf = call_name.split(".")[-1].lower()
        if leaf in {"reference", "ref_forward", "torch_forward", "get_inputs"}:
            return f"calls_suspicious_function:{call_name}"
        if "reference" in leaf or leaf.startswith("ref_"):
            return f"calls_suspicious_function:{call_name}"
        if call_name.startswith("self.") and leaf not in _ALLOWED_TENSOR_METHODS:
            return f"calls_module_from_forward:{call_name}"
    return None


def _find_torch_fallback(forward: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    argument_names = _direct_argument_aliases(forward)
    for node in ast.walk(forward):
        if isinstance(node, ast.Subscript):
            value_name = _name_of(node.value)
            root = value_name.split(".")[0]
            leaf = value_name.split(".")[-1]
            if root in argument_names and leaf not in _ALLOWED_TENSOR_ATTRIBUTES:
                return f"obvious_torch_fallback:tensor_subscript:{value_name}"
        if isinstance(node, ast.Attribute):
            attribute_name = _name_of(node)
            leaf = attribute_name.split(".")[-1]
            if leaf in _TORCH_COMPUTE_METHODS and (
                attribute_name.startswith("torch.")
                or attribute_name.startswith("F.")
                or attribute_name.startswith("nn.functional.")
            ):
                return f"obvious_torch_fallback:{attribute_name}"
            root = attribute_name.split(".")[0]
            if (
                root in argument_names
                and leaf not in _ALLOWED_TENSOR_METHODS
                and leaf not in _ALLOWED_TENSOR_ATTRIBUTES
            ):
                return f"obvious_torch_fallback:tensor_attribute:{attribute_name}"
        if isinstance(node, ast.Call):
            call_name = _name_of(node.func)
            leaf = call_name.split(".")[-1]
            if call_name.startswith("torch.") and call_name not in _ALLOWED_TORCH_CALLS:
                return f"obvious_torch_fallback:{call_name}"
            if call_name.startswith("F.") or call_name.startswith("nn.functional."):
                return f"obvious_torch_fallback:{call_name}"
            if leaf in _TORCH_COMPUTE_METHODS and leaf not in _ALLOWED_TENSOR_METHODS:
                return f"obvious_torch_fallback:{call_name}"
            root = call_name.split(".")[0]
            if root in argument_names and leaf not in _ALLOWED_TENSOR_METHODS:
                return f"obvious_torch_fallback:tensor_method:{call_name}"
        if isinstance(node, ast.BinOp) and _references_any_name(node, argument_names):
            operator = {
                ast.Add: "direct_add",
                ast.Mult: "direct_mul",
                ast.MatMult: "direct_matmul",
                ast.Sub: "direct_sub",
                ast.Div: "direct_div",
            }.get(type(node.op))
            if operator:
                return f"obvious_torch_fallback:{operator}"
        if isinstance(node, ast.UnaryOp) and _references_any_name(node, argument_names):
            return f"obvious_torch_fallback:{type(node.op).__name__.lower()}"
    return None


def _direct_argument_aliases(
    forward: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    """Track simple ``alias = input`` bindings used to hide tensor operations."""

    argument_nodes = [*forward.args.posonlyargs, *forward.args.args, *forward.args.kwonlyargs]
    if forward.args.vararg is not None:
        argument_nodes.append(forward.args.vararg)
    if forward.args.kwarg is not None:
        argument_nodes.append(forward.args.kwarg)
    names = {arg.arg for arg in argument_nodes if arg.arg not in {"self", "cls"}}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(forward):
            value = getattr(node, "value", None)
            if not isinstance(value, ast.Name) or value.id not in names:
                continue
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in names:
                    names.add(target.id)
                    changed = True
    return names


def _references_any_name(node: ast.AST, names: set[str]) -> bool:
    return any(isinstance(child, ast.Name) and child.id in names for child in ast.walk(node))


def _name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_of(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    return ""
