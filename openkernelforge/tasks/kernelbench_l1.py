"""KernelBench L1 adapter skeleton.

This module intentionally keeps the first KernelBench integration narrow: load
local task files, validate references and inputs, and provide metadata for the
rigorous benchmark path. It does not download KernelBench or invoke LLMs.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from openkernelforge.tasks.base import KernelTask, Shape, TaskTolerance


SETUP_INSTRUCTIONS = (
    "KernelBench L1 tasks were not found. Pass --kernelbench-dir pointing to a "
    "local KernelBench checkout or a directory of KernelBench-style Python task "
    "files. This command does not download KernelBench automatically."
)


class KernelBenchL1Error(RuntimeError):
    """Raised when KernelBench task loading cannot proceed."""


@dataclass(frozen=True)
class KernelBenchCandidateProvider:
    mode: str = "none"
    root: str | None = None

    def candidate_for_task(self, task_id: str) -> Path | None:
        if self.mode == "none":
            return None
        if self.mode == "llm_later":
            raise NotImplementedError(
                "candidate_provider=llm_later is a placeholder. KernelBench L1 "
                "LLM generation is intentionally not implemented in this sprint."
            )
        if self.mode == "existing_file":
            if not self.root:
                raise KernelBenchL1Error("candidate_provider=existing_file requires candidate_root")
            root = Path(self.root)
            candidates = [
                root / f"{task_id}.py",
                root / task_id / "candidate.py",
                root / task_id / "forward.py",
            ]
            for path in candidates:
                if path.exists():
                    return path
            return None
        raise KernelBenchL1Error(f"Unknown KernelBench candidate_provider: {self.mode}")


def make_candidate_provider(config: dict[str, Any] | None = None) -> KernelBenchCandidateProvider:
    data = config or {}
    return KernelBenchCandidateProvider(
        mode=str(data.get("candidate_provider", "none")),
        root=data.get("candidate_root"),
    )


def load_kernelbench_l1_tasks(
    kernelbench_dir: str | Path,
    *,
    task_ids: list[str] | None = None,
    max_tasks: int | None = None,
) -> list[KernelTask]:
    """Load a small KernelBench L1 task subset from local Python files."""

    root = Path(kernelbench_dir).expanduser()
    if not root.exists():
        raise KernelBenchL1Error(f"{SETUP_INSTRUCTIONS}\nMissing path: {root}")
    if not root.is_dir():
        raise KernelBenchL1Error(f"KernelBench path is not a directory: {root}")

    selected_ids = set(task_ids or [])
    tasks: list[KernelTask] = []
    errors: list[str] = []
    for path in _discover_task_files(root):
        try:
            task = _task_from_file(root, path)
        except Exception as exc:  # pragma: no cover - exercised through error reporting
            errors.append(f"{path.relative_to(root)}: {exc}")
            continue
        if selected_ids and task.task_id not in selected_ids:
            continue
        tasks.append(task)
        if max_tasks is not None and len(tasks) >= max_tasks:
            break

    if not tasks:
        details = "\n".join(f"- {error}" for error in errors[:10])
        suffix = f"\nTask load errors:\n{details}" if details else ""
        raise KernelBenchL1Error(
            f"{SETUP_INSTRUCTIONS}\nNo loadable L1 task files found under: {root}{suffix}"
        )
    return tasks


def _discover_task_files(root: Path) -> list[Path]:
    files = [path for path in sorted(root.rglob("*.py")) if not path.name.startswith("__")]
    preferred = [
        path
        for path in files
        if "level1" in path.as_posix().lower()
        or "level_1" in path.as_posix().lower()
        or "/l1/" in path.as_posix().lower()
        or "l1" in path.stem.lower()
    ]
    return preferred or files


def _task_from_file(root: Path, path: Path) -> KernelTask:
    module = _load_module(path)
    rel_id = path.relative_to(root).with_suffix("").as_posix().replace("/", "__")
    task_id = str(getattr(module, "TASK_ID", getattr(module, "task_id", rel_id)))
    op_family = str(getattr(module, "OP_FAMILY", getattr(module, "op_family", "kernelbench_l1")))
    shape = _extract_shape(module)
    tolerance = _extract_tolerance(module)
    reference_fn, input_generator = _extract_reference_and_generator(module, shape)
    source = _safe_source(reference_fn)
    if source is None:
        source = f"# Loaded from {path}"
    return KernelTask(
        task_id=task_id,
        name=str(getattr(module, "TASK_NAME", getattr(module, "name", task_id))),
        description=str(getattr(module, "DESCRIPTION", getattr(module, "description", op_family))),
        reference_fn=reference_fn,
        input_generator=input_generator,
        allowed_dtypes=(torch.float32,),
        tolerance=tolerance,
        benchmark_shapes=[shape],
        reference_source=source,
        metadata={
            "task_family": "kernelbench_l1",
            "op_family": op_family,
            "input_spec": getattr(module, "INPUT_SPEC", getattr(module, "input_spec", {})),
            "shape_metadata": _shape_metadata(shape),
            "source_path": str(path),
            "source_relative_path": path.relative_to(root).as_posix(),
        },
    )


def _load_module(path: Path) -> Any:
    module_name = f"_okf_kernelbench_l1_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise KernelBenchL1Error(f"Could not import task file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _extract_shape(module: Any) -> Shape:
    raw = getattr(module, "BENCHMARK_SHAPES", getattr(module, "benchmark_shapes", None))
    if raw is None:
        raw = getattr(module, "SHAPES", getattr(module, "shapes", None))
    if raw:
        first = raw[0] if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)) else raw
        return tuple(int(item) for item in first)
    input_spec = getattr(module, "INPUT_SPEC", getattr(module, "input_spec", {})) or {}
    if isinstance(input_spec, dict) and input_spec.get("shape"):
        return tuple(int(item) for item in input_spec["shape"])
    if hasattr(module, "get_inputs"):
        inputs = module.get_inputs()
        if isinstance(inputs, torch.Tensor):
            return tuple(int(dim) for dim in inputs.shape)
        if isinstance(inputs, (list, tuple)) and inputs and isinstance(inputs[0], torch.Tensor):
            return tuple(int(dim) for dim in inputs[0].shape)
    return (128, 128)


def _extract_tolerance(module: Any) -> TaskTolerance:
    raw = getattr(module, "TOLERANCE", getattr(module, "tolerance", None))
    if isinstance(raw, TaskTolerance):
        return raw
    if isinstance(raw, dict):
        return TaskTolerance(rtol=float(raw.get("rtol", 1e-4)), atol=float(raw.get("atol", 1e-5)))
    return TaskTolerance(rtol=1e-4, atol=1e-5)


def _extract_reference_and_generator(
    module: Any,
    shape: Shape,
) -> tuple[Callable[..., Any], Callable[[int, Shape, torch.dtype, torch.device], tuple[Any, ...]]]:
    reference = getattr(module, "reference_fn", getattr(module, "reference", None))
    generator = getattr(module, "input_generator", getattr(module, "generate_inputs", None))
    if callable(reference) and callable(generator):
        return reference, _wrap_custom_generator(generator)

    model_cls = getattr(module, "Model", None)
    get_inputs = getattr(module, "get_inputs", None)
    if model_cls is not None and callable(get_inputs):
        init_inputs_fn = getattr(module, "get_init_inputs", None)

        def reference_from_model(*args: Any) -> Any:
            init_args = init_inputs_fn() if callable(init_inputs_fn) else []
            if init_args is None:
                init_args = []
            if not isinstance(init_args, (list, tuple)):
                init_args = [init_args]
            model = model_cls(*init_args)
            if hasattr(model, "eval"):
                model = model.eval()
            if args and isinstance(args[0], torch.Tensor):
                model = model.to(device=args[0].device, dtype=args[0].dtype)
            with torch.no_grad():
                return model(*args)

        def model_generator(
            seed: int,
            selected_shape: Shape,
            dtype: torch.dtype,
            device: torch.device,
        ) -> tuple[Any, ...]:
            torch.manual_seed(seed)
            inputs = get_inputs()
            if not isinstance(inputs, (list, tuple)):
                inputs = (inputs,)
            return tuple(_move_value(value, dtype, device) for value in inputs)

        return reference_from_model, model_generator

    raise KernelBenchL1Error(
        "Task file must define reference_fn/reference plus input_generator/generate_inputs, "
        "or KernelBench-style Model and get_inputs functions."
    )


def _wrap_custom_generator(generator: Callable[..., Any]) -> Callable[[int, Shape, torch.dtype, torch.device], tuple[Any, ...]]:
    def wrapped(seed: int, shape: Shape, dtype: torch.dtype, device: torch.device) -> tuple[Any, ...]:
        value = generator(seed, shape, dtype, device)
        if not isinstance(value, tuple):
            value = tuple(value) if isinstance(value, list) else (value,)
        return tuple(value)

    return wrapped


def _move_value(value: Any, dtype: torch.dtype, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, (list, tuple)):
        return type(value)(_move_value(item, dtype, device) for item in value)
    return value


def _shape_metadata(shape: Shape) -> dict[str, Any]:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    metadata: dict[str, Any] = {"shape": list(shape), "rank": len(shape), "numel": numel}
    if len(shape) >= 2:
        metadata["rows"] = int(shape[0])
        metadata["feature_dim"] = int(shape[-1])
    return metadata


def _safe_source(fn: Callable[..., Any]) -> str | None:
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return None
