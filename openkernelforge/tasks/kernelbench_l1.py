"""KernelBench L1 task loading, contract binding, and memory preflight.

This module loads a local checkout and adapts official ``Model`` tasks to the
``ModelNew`` contract. It does not download KernelBench or invoke model APIs.
"""

from __future__ import annotations

import importlib.util
import hashlib
import inspect
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
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


class KernelBenchModelReference:
    """Persistent reference model for an official KernelBench task.

    KernelBench task files define ``Model`` and ``get_init_inputs``.  The
    generated implementation is expected to define ``ModelNew`` with the same
    constructor contract.  Reference and candidate modules are initialized
    under the same deterministic seed and are moved to the target device
    before verification or timing begins.
    """

    init_seed = 0

    def __init__(
        self,
        model_cls: type,
        init_inputs_fn: Callable[[], Any] | None,
    ) -> None:
        self.model_cls = model_cls
        self.init_inputs_fn = init_inputs_fn
        self.init_args = _materialize_init_args(init_inputs_fn, seed=self.init_seed)
        self._models: dict[tuple[str, str], torch.nn.Module] = {}
        cpu_model = self._instantiate(model_cls, torch.float32, torch.device("cpu"))
        self._models[("cpu", "torch.float32")] = cpu_model
        self.has_model_state = bool(cpu_model.state_dict())
        self.state_keys = tuple(cpu_model.state_dict().keys())
        self.state_bytes = sum(
            int(value.numel() * value.element_size())
            for value in cpu_model.state_dict().values()
            if isinstance(value, torch.Tensor)
        )

    @property
    def candidate_contract(self) -> str:
        return "model_new"

    def prepare_for(
        self,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.nn.Module:
        selected_device = torch.device(device)
        key = (str(selected_device), str(dtype))
        model = self._models.get(key)
        if model is None:
            model = self._instantiate(self.model_cls, dtype, selected_device)
            self._models[key] = model
        return model

    def bind_candidate(
        self,
        module: ModuleType,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> Callable[..., Any]:
        model_new = getattr(module, "ModelNew", None)
        if isinstance(model_new, type):
            candidate = KernelBenchModelCandidate(self, model_new)
            candidate.prepare_for(dtype, device)
            return candidate

        raise KernelBenchL1Error(
            "Official KernelBench Model tasks require candidate class ModelNew with the same "
            "get_init_inputs() constructor contract; free forward(*args) candidates are unsupported"
        )

    def __call__(self, *args: Any) -> Any:
        dtype, device = _call_dtype_device(args)
        model = self.prepare_for(dtype, device)
        with torch.no_grad():
            return model(*args)

    def reconstruct_per_call(self, *args: Any) -> Any:
        """Historical contaminated lifecycle retained only for evaluator ablation."""

        dtype, device = _call_dtype_device(args)
        model = self._instantiate(self.model_cls, dtype, device)
        with torch.no_grad():
            return model(*args)

    def reconstruct_per_call_profiled(self, *args: Any) -> tuple[Any, dict[str, float]]:
        """Profile lifecycle components in a control-only synchronized call.

        This intentionally adds synchronization boundaries and therefore does
        not replace the separate exact end-to-end contaminated-call timing.
        It only decomposes where host time is spent.
        """

        dtype, device = _call_dtype_device(args)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_started = time.perf_counter()
        init_started = time.perf_counter()
        init_args = tuple(
            _move_value(value, dtype, device)
            for value in _clone_value(self.init_args)
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        init_completed = time.perf_counter()
        cuda_devices: list[int] = []
        if device.type == "cuda" and torch.cuda.is_available():
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.init_seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(self.init_seed)
            model = self.model_cls(*init_args)
        constructor_completed = time.perf_counter()
        if not isinstance(model, torch.nn.Module):
            raise KernelBenchL1Error(
                f"KernelBench model class must construct torch.nn.Module, got {type(model)!r}"
            )
        model = model.eval().to(device=device, dtype=dtype)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        transfer_completed = time.perf_counter()
        start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if start_event is not None:
            start_event.record()
        with torch.no_grad():
            output = model(*args)
        if end_event is not None:
            end_event.record()
        enqueue_completed = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        completed = time.perf_counter()
        return output, {
            "init_argument_materialization_ms": (init_completed - init_started) * 1000.0,
            "constructor_host_ms": (constructor_completed - init_completed) * 1000.0,
            "state_device_transfer_ms": (transfer_completed - constructor_completed) * 1000.0,
            "forward_enqueue_host_ms": (enqueue_completed - transfer_completed) * 1000.0,
            "forward_synchronization_ms": (completed - enqueue_completed) * 1000.0,
            "forward_cuda_event_ms": (
                float(start_event.elapsed_time(end_event))
                if start_event is not None and end_event is not None
                else 0.0
            ),
            "profiled_total_host_ms": (completed - total_started) * 1000.0,
        }

    def _instantiate(
        self,
        model_cls: type,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.nn.Module:
        init_args = tuple(
            _move_value(value, dtype, device)
            for value in _clone_value(self.init_args)
        )
        cuda_devices: list[int] = []
        if device.type == "cuda" and torch.cuda.is_available():
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.init_seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(self.init_seed)
            model = model_cls(*init_args)
        if not isinstance(model, torch.nn.Module):
            raise KernelBenchL1Error(
                f"KernelBench model class must construct torch.nn.Module, got {type(model)!r}"
            )
        model = model.eval()
        return model.to(device=device, dtype=dtype)


class KernelBenchModelCandidate:
    """Persistent ``ModelNew`` instances initialized like the reference."""

    def __init__(self, reference: KernelBenchModelReference, model_cls: type) -> None:
        self.reference = reference
        self.model_cls = model_cls
        self._models: dict[tuple[str, str], torch.nn.Module] = {}

    def prepare_for(
        self,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.nn.Module:
        selected_device = torch.device(device)
        key = (str(selected_device), str(dtype))
        model = self._models.get(key)
        if model is None:
            model = self.reference._instantiate(self.model_cls, dtype, selected_device)
            self._models[key] = model
        return model

    def __call__(self, *args: Any) -> Any:
        dtype, device = _call_dtype_device(args)
        return self.prepare_for(dtype, device)(*args)


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
        if self.mode in {"gemini", "gemini_repair"}:
            return None
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
    stratify_by_family: bool = False,
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
    files = _discover_task_files(root)
    if stratify_by_family:
        files = _stratify_task_files(files)
    for path in files:
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


def _stratify_task_files(files: list[Path]) -> list[Path]:
    buckets: dict[str, list[Path]] = {}
    for path in files:
        buckets.setdefault(_infer_op_family(path), []).append(path)
    ordered: list[Path] = []
    family_names = sorted(buckets)
    while any(buckets.values()):
        for family in family_names:
            bucket = buckets[family]
            if bucket:
                ordered.append(bucket.pop(0))
    return ordered


def _task_from_file(root: Path, path: Path) -> KernelTask:
    module = _load_module(root, path)
    rel_id = path.relative_to(root).with_suffix("").as_posix().replace("/", "__")
    task_id = str(getattr(module, "TASK_ID", getattr(module, "task_id", rel_id)))
    op_family = str(getattr(module, "OP_FAMILY", getattr(module, "op_family", _infer_op_family(path))))
    shape = _extract_shape(module)
    tolerance = _extract_tolerance(module)
    reference_fn, input_generator, contract_metadata = _extract_reference_and_generator(module, shape)
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
            "module_name": getattr(module, "__name__", None),
            **contract_metadata,
        },
    )


def _load_module(root: Path, path: Path) -> Any:
    module_name = _stable_module_name(root, path)
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise KernelBenchL1Error(f"Could not import task file: {path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = ""
    sys.modules[module_name] = module
    added_paths: list[str] = []
    for candidate in (root, path.parent):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
            added_paths.append(text)
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        for text in reversed(added_paths):
            try:
                sys.path.remove(text)
            except ValueError:  # pragma: no cover - defensive cleanup
                pass
    return module


def _stable_module_name(root: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = path.resolve()
    stem = rel.with_suffix("").as_posix()
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", stem).strip("_")
    if not safe or safe[0].isdigit():
        safe = f"m_{safe}"
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"openkernelforge_kernelbench_l1_{safe}_{digest}"


def _infer_op_family(path: Path) -> str:
    text = path.stem.lower()
    if any(token in text for token in ("conv", "convolution")):
        return "convolution"
    if any(token in text for token in ("matmul", "matrix", "bmm", "gemm")):
        return "matmul"
    if any(token in text for token in ("norm", "batchnorm", "layernorm", "groupnorm")):
        return "normalization"
    if any(token in text for token in ("pool", "avgpool", "maxpool")):
        return "pooling"
    if any(token in text for token in ("loss", "hinge", "crossentropy", "mse")):
        return "loss"
    if any(token in text for token in ("softmax", "relu", "gelu", "sigmoid", "tanh", "swish", "selu", "elu")):
        return "activation"
    if any(token in text for token in ("sum", "mean", "prod", "amax", "amin", "reduce")):
        return "reduction"
    if any(token in text for token in ("cumsum", "cumprod", "scan")):
        return "scan"
    return "kernelbench_l1"


def estimate_input_memory(value: Any) -> dict[str, Any]:
    """Measure input tensor bytes without estimating outputs or workspace."""

    seen: set[int] = set()
    tensors: list[dict[str, Any]] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            ident = id(item)
            if ident in seen:
                return
            seen.add(ident)
            numel = int(item.numel())
            bytes_ = int(numel * item.element_size())
            tensors.append(
                {
                    "path": path,
                    "shape": list(item.shape),
                    "dtype": str(item.dtype).replace("torch.", ""),
                    "device": str(item.device),
                    "numel": numel,
                    "bytes": bytes_,
                }
            )
            return
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "inputs")
    total_bytes = sum(item["bytes"] for item in tensors)
    max_numel = max((item["numel"] for item in tensors), default=0)
    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "max_tensor_numel": max_numel,
        "tensor_count": len(tensors),
        "tensors": tensors,
        "estimate_scope": "input_tensors_only",
    }


def estimate_task_memory(
    task: KernelTask,
    inputs: Any,
    *,
    known_overhead_bytes: int = 0,
) -> dict[str, Any]:
    """Estimate known peak residency for the verification lifecycle.

    Verification holds the original input tree, independent reference and
    candidate copies, and before-call snapshots for mutation checking. It also
    keeps reference and candidate module state resident. Outputs, compiler
    workspace, temporary tensors, and allocator fragmentation remain unknown,
    so this is still a lower bound rather than a guaranteed peak.
    """

    estimate = estimate_input_memory(inputs)
    reference_state_bytes = int(getattr(task.reference_fn, "state_bytes", 0) or 0)
    verification_input_copy_factor = 5
    model_state_copy_factor = 2 if reference_state_bytes else 0
    minimum_resident_bytes = (
        int(estimate["total_bytes"]) + reference_state_bytes + max(0, int(known_overhead_bytes))
    )
    estimated_known_peak_bytes = (
        verification_input_copy_factor * int(estimate["total_bytes"])
        + model_state_copy_factor * reference_state_bytes
        + max(0, int(known_overhead_bytes))
    )
    estimate.update(
        {
            "reference_state_bytes": reference_state_bytes,
            "known_overhead_bytes": max(0, int(known_overhead_bytes)),
            "minimum_resident_bytes": minimum_resident_bytes,
            "minimum_resident_mb": minimum_resident_bytes / (1024 * 1024),
            "verification_input_copy_factor": verification_input_copy_factor,
            "model_state_copy_factor": model_state_copy_factor,
            "estimated_known_peak_bytes": estimated_known_peak_bytes,
            "estimated_known_peak_mb": estimated_known_peak_bytes / (1024 * 1024),
            "estimate_scope": (
                "verification_input_copies_plus_reference_and_candidate_state_plus_known_overhead; "
                "excludes outputs, temporary tensors, compiler workspace, and allocator fragmentation"
            ),
        }
    )
    return estimate


def generate_inputs_for_memory_estimate(
    task: KernelTask,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    allow_cpu_fallback: bool = False,
) -> tuple[Any, ...]:
    """Prefer metadata-only inputs so the memory cap runs before real allocation."""

    metadata_generator = getattr(task.input_generator, "_okf_metadata_generator", None)
    if callable(metadata_generator):
        try:
            return metadata_generator(seed, task.benchmark_shapes[0], dtype)
        except Exception as exc:
            if not allow_cpu_fallback:
                raise KernelBenchL1Error(
                    "metadata-only get_inputs() failed; CPU materialization is disabled"
                ) from exc
    return task.generate_inputs(
        seed,
        task.benchmark_shapes[0],
        dtype,
        torch.device("cpu"),
    )


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
        try:
            inputs = _materialize_get_inputs(module.get_inputs, seed=0, metadata_only=True)
        except KernelBenchL1Error:
            return ()
        if isinstance(inputs, torch.Tensor):
            return tuple(int(dim) for dim in inputs.shape)
        if isinstance(inputs, (list, tuple)) and inputs and isinstance(inputs[0], torch.Tensor):
            return tuple(int(dim) for dim in inputs[0].shape)
    return ()


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
) -> tuple[
    Callable[..., Any],
    Callable[[int, Shape, torch.dtype, torch.device], tuple[Any, ...]],
    dict[str, Any],
]:
    reference = getattr(module, "reference_fn", getattr(module, "reference", None))
    generator = getattr(module, "input_generator", getattr(module, "generate_inputs", None))
    if callable(reference) and callable(generator):
        return reference, _wrap_custom_generator(generator), {
            "candidate_contract": "forward",
            "reference_has_model_state": False,
            "reference_state_keys": [],
        }

    model_cls = getattr(module, "Model", None)
    get_inputs = getattr(module, "get_inputs", None)
    if model_cls is not None and callable(get_inputs):
        init_inputs_fn = getattr(module, "get_init_inputs", None)
        reference_from_model = KernelBenchModelReference(
            model_cls,
            init_inputs_fn if callable(init_inputs_fn) else None,
        )

        def model_generator(
            seed: int,
            selected_shape: Shape,
            dtype: torch.dtype,
            device: torch.device,
        ) -> tuple[Any, ...]:
            cuda_devices: list[int] = []
            if device.type == "cuda" and torch.cuda.is_available():
                cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                if cuda_devices:
                    torch.cuda.manual_seed_all(seed)
                inputs = get_inputs()
            if not isinstance(inputs, (list, tuple)):
                inputs = (inputs,)
            return tuple(_move_value(value, dtype, device) for value in inputs)

        def model_metadata_generator(
            seed: int,
            selected_shape: Shape,
            dtype: torch.dtype,
        ) -> tuple[Any, ...]:
            del selected_shape
            inputs = _materialize_get_inputs(get_inputs, seed=seed, metadata_only=True)
            if not isinstance(inputs, (list, tuple)):
                inputs = (inputs,)
            return tuple(_move_metadata_value(value, dtype) for value in inputs)

        setattr(model_generator, "_okf_metadata_generator", model_metadata_generator)

        return reference_from_model, model_generator, {
            "candidate_contract": reference_from_model.candidate_contract,
            "reference_has_model_state": reference_from_model.has_model_state,
            "reference_state_keys": list(reference_from_model.state_keys),
            "model_init_seed": reference_from_model.init_seed,
        }

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
    if isinstance(value, dict):
        return {key: _move_value(item, dtype, device) for key, item in value.items()}
    return value


def _move_metadata_value(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if value.is_floating_point() else value
    if isinstance(value, list):
        return [_move_metadata_value(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_metadata_value(item, dtype) for item in value)
    if isinstance(value, dict):
        return {key: _move_metadata_value(item, dtype) for key, item in value.items()}
    return value


def _materialize_get_inputs(
    get_inputs: Callable[[], Any],
    *,
    seed: int,
    metadata_only: bool,
) -> Any:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if metadata_only:
            try:
                with torch.device("meta"):
                    return get_inputs()
            except Exception as exc:
                raise KernelBenchL1Error("get_inputs() does not support metadata-only execution") from exc
        return get_inputs()


def bind_kernelbench_candidate(
    task: KernelTask,
    module: ModuleType,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> Callable[..., Any]:
    """Bind a loaded candidate module to the task's declared contract."""

    reference = task.reference_fn
    if isinstance(reference, KernelBenchModelReference):
        reference.prepare_for(dtype, device)
        return reference.bind_candidate(module, dtype, device)
    forward = getattr(module, "forward", None)
    if not callable(forward):
        raise KernelBenchL1Error("Candidate must expose callable forward(*args)")
    return forward


def _materialize_init_args(
    init_inputs_fn: Callable[[], Any] | None,
    *,
    seed: int,
) -> tuple[Any, ...]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        value = init_inputs_fn() if callable(init_inputs_fn) else []
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _call_dtype_device(args: tuple[Any, ...]) -> tuple[torch.dtype, torch.device]:
    tensors = _collect_tensors(args)
    if not tensors:
        return torch.float32, torch.device("cpu")
    first = tensors[0]
    dtype = first.dtype if first.is_floating_point() else torch.float32
    return dtype, first.device


def _collect_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        tensors: list[torch.Tensor] = []
        for child in value.values():
            tensors.extend(_collect_tensors(child))
        return tensors
    if isinstance(value, (list, tuple)):
        tensors = []
        for child in value:
            tensors.extend(_collect_tensors(child))
        return tensors
    return []


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_clone_value(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(child) for child in value)
    return value


def _shape_metadata(shape: Shape) -> dict[str, Any]:
    if not shape:
        return {
            "shape": [],
            "rank": None,
            "numel": None,
            "inference_status": "unavailable_without_input_materialization",
        }
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
