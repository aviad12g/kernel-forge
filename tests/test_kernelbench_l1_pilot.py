from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from openkernelforge.cli import main
from openkernelforge.config import load_config
from openkernelforge.harness.benchmarker import BenchmarkResult, RuntimeStats
from openkernelforge.reports.kernelbench_l1 import (
    _load_repair_index,
    _validate_repair_contracts,
    run_kernelbench_l1_check,
    write_kernelbench_l1_report,
)
from openkernelforge.tasks.kernelbench_l1 import (
    bind_kernelbench_candidate,
    estimate_input_memory,
    estimate_task_memory,
    generate_inputs_for_memory_estimate,
    KernelBenchL1Error,
    load_kernelbench_l1_tasks,
    make_candidate_provider,
)


def _write_synthetic_task(root: Path, task_id: str = "kb_l1_add") -> Path:
    task_dir = root / "level1"
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"{task_id}.py"
    path.write_text(
        f"""
import torch

TASK_ID = "{task_id}"
TASK_NAME = "Synthetic add"
OP_FAMILY = "elementwise"
BENCHMARK_SHAPES = [(4, 4)]
TOLERANCE = {{"rtol": 1e-4, "atol": 1e-5}}
INPUT_SPEC = {{"shape": [4, 4], "inputs": ["x", "y"]}}

def reference_fn(x, y):
    return x + y

def input_generator(seed, shape, dtype, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    y = torch.randn(shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    return x, y
""",
        encoding="utf-8",
    )
    return path


def _write_parameterized_model_task(root: Path, task_id: str = "kb_l1_weighted") -> Path:
    task_dir = root / "level1"
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"{task_id}.py"
    path.write_text(
        f'''
import torch
from torch import nn

TASK_ID = "{task_id}"
TASK_NAME = "Synthetic weighted op"
OP_FAMILY = "elementwise"
BENCHMARK_SHAPES = [(4,)]
CONSTRUCTIONS = 0

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        global CONSTRUCTIONS
        CONSTRUCTIONS += 1
        self.weight = nn.Parameter(torch.randn(4))

    def forward(self, x):
        return x * self.weight

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(4)]
''',
        encoding="utf-8",
    )
    return path


def _write_init_input_model_task(root: Path, task_id: str = "kb_l1_init_input") -> Path:
    task_dir = root / "level1"
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"{task_id}.py"
    path.write_text(
        f'''
import torch
from torch import nn

TASK_ID = "{task_id}"
TASK_NAME = "Synthetic init-input op"
OP_FAMILY = "elementwise"
BENCHMARK_SHAPES = [(4,)]
INIT_CALLS = 0

class Model(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

def get_init_inputs():
    global INIT_CALLS
    INIT_CALLS += 1
    return [torch.rand(())]

def get_inputs():
    return [torch.randn(4)]
''',
        encoding="utf-8",
    )
    return path


def _write_metadata_only_large_task(root: Path, task_id: str = "kb_l1_large_meta") -> Path:
    task_dir = root / "level1"
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"{task_id}.py"
    path.write_text(
        f'''
import torch
from torch import nn

TASK_ID = "{task_id}"
TASK_NAME = "Synthetic metadata-only large op"
OP_FAMILY = "activation"

class Model(nn.Module):
    def forward(self, x):
        return x

def get_init_inputs():
    return []

def get_inputs():
    x = torch.empty(1_000_000_000)
    if x.device.type != "meta":
        raise RuntimeError("real allocation path should not run during preflight")
    return [x]
''',
        encoding="utf-8",
    )
    return path


def _write_meta_unsupported_task(root: Path, task_id: str = "kb_l1_meta_unsupported") -> Path:
    task_dir = root / "level1"
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"{task_id}.py"
    path.write_text(
        f'''
import torch
from torch import nn

TASK_ID = "{task_id}"
TASK_NAME = "Synthetic metadata-unsupported op"
OP_FAMILY = "activation"

class Model(nn.Module):
    def forward(self, x):
        return x

def get_init_inputs():
    return []

def get_inputs():
    x = torch.empty(4)
    if x.device.type == "meta":
        raise RuntimeError("synthetic meta failure")
    raise AssertionError("CPU fallback must not run unless explicitly enabled")
''',
        encoding="utf-8",
    )
    return path


def _write_synthetic_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
tasks: []
output_dir: {json.dumps(str(output_dir))}
kernelbench:
  max_tasks: 5
  candidate_provider: none
benchmark:
  enabled: true
  timing_mode: wall_clock
  warmup: 0
  repeat: 2
  independent_sessions: 1
  include_torch_compile: false
execution:
  require_cuda: false
  require_triton: false
  require_tiny_triton_kernel: false
""",
        encoding="utf-8",
    )


def _write_synthetic_gemini_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
tasks: []
output_dir: {json.dumps(str(output_dir))}
kernelbench:
  max_tasks: 1
  candidate_provider: gemini
  candidates_per_task: 1
  record_skipped_tasks: true
agent:
  type: llm
  backend: fake
  model: test-gemini
  api_key_env: GEMINI_API_KEY
  allow_torch_fallback: true
  max_attempts: 1
  candidates_per_attempt: 1
benchmark:
  enabled: true
  timing_mode: wall_clock
  warmup: 0
  repeat: 2
  independent_sessions: 1
  include_torch_compile: false
execution:
  require_cuda: false
  require_triton: false
  require_tiny_triton_kernel: false
""",
        encoding="utf-8",
    )


def _write_synthetic_gemini_repair_config(
    path: Path,
    output_dir: Path,
    taxonomy_path: Path,
) -> None:
    path.write_text(
        f"""
tasks: []
output_dir: {json.dumps(str(output_dir))}
kernelbench:
  max_tasks: 1
  candidate_provider: gemini_repair
  candidates_per_task: 1
  repair_taxonomy_path: {json.dumps(str(taxonomy_path))}
  repair_task_ids:
    - kb_l1_add
agent:
  type: llm
  backend: fake
  model: test-gemini
  api_key_env: GEMINI_API_KEY
  allow_torch_fallback: true
  max_attempts: 1
  candidates_per_attempt: 1
benchmark:
  enabled: true
  timing_mode: wall_clock
  warmup: 0
  repeat: 2
  independent_sessions: 1
  include_torch_compile: false
execution:
  require_cuda: false
  require_triton: false
  require_tiny_triton_kernel: false
""",
        encoding="utf-8",
    )


def _write_oversized_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        f"""
tasks: []
output_dir: {json.dumps(str(output_dir))}
kernelbench:
  max_tasks: 1
  candidate_provider: none
  max_numel_per_input: 8
  record_skipped_tasks: true
benchmark:
  enabled: true
  timing_mode: wall_clock
  warmup: 0
  repeat: 1
  independent_sessions: 1
  include_torch_compile: false
execution:
  require_cuda: false
  require_triton: false
  require_tiny_triton_kernel: false
""",
        encoding="utf-8",
    )


def _write_existing_file_config(path: Path, output_dir: Path, candidate_root: Path) -> None:
    path.write_text(
        f"""
tasks: []
output_dir: {json.dumps(str(output_dir))}
kernelbench:
  max_tasks: 1
  candidate_provider: existing_file
  candidate_root: {json.dumps(str(candidate_root))}
agent:
  allow_torch_fallback: true
benchmark:
  enabled: true
  timing_mode: wall_clock
  warmup: 0
  repeat: 2
  independent_sessions: 1
  include_torch_compile: false
execution:
  require_cuda: false
  require_triton: false
  require_tiny_triton_kernel: false
""",
        encoding="utf-8",
    )


def test_kernelbench_l1_adapter_loads_synthetic_task_stubs(tmp_path):
    _write_synthetic_task(tmp_path)
    tasks = load_kernelbench_l1_tasks(tmp_path, max_tasks=5)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.task_id == "kb_l1_add"
    assert task.metadata["task_family"] == "kernelbench_l1"
    assert task.metadata["op_family"] == "elementwise"
    assert task.metadata["module_name"] in sys.modules
    inputs = task.generate_inputs(0)
    assert len(inputs) == 2
    assert task.reference_fn(*inputs).shape == inputs[0].shape


def test_kernelbench_l1_dynamic_loader_uses_stable_importable_module_name(tmp_path):
    _write_synthetic_task(tmp_path, "kb_l1_stable")
    task = load_kernelbench_l1_tasks(tmp_path, max_tasks=1)[0]
    module_name = task.metadata["module_name"]
    assert module_name.startswith("openkernelforge_kernelbench_l1_")
    assert "_okf_kernelbench_l1_" not in module_name
    assert module_name in sys.modules
    assert getattr(sys.modules[module_name], "__file__", "").endswith("kb_l1_stable.py")


def test_kernelbench_parameterized_task_uses_persistent_modelnew_contract(tmp_path):
    _write_parameterized_model_task(tmp_path)
    task = load_kernelbench_l1_tasks(tmp_path, max_tasks=1)[0]
    task_module = sys.modules[task.metadata["module_name"]]

    assert task.metadata["candidate_contract"] == "model_new"
    assert task.metadata["reference_has_model_state"] is True
    inputs = task.generate_inputs(0)
    first = task.reference_fn(*inputs)
    second = task.reference_fn(*inputs)
    assert torch.equal(first, second)
    assert task_module.CONSTRUCTIONS == 1
    reconstructed = task.reference_fn.reconstruct_per_call(*inputs)
    assert torch.equal(reconstructed, first)
    assert task_module.CONSTRUCTIONS == 2
    assert torch.equal(task.reference_fn(*inputs), first)
    assert task_module.CONSTRUCTIONS == 2

    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text(
        """
import torch
from torch import nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4))

    def forward(self, x):
        return x * self.weight
""",
        encoding="utf-8",
    )
    from openkernelforge.harness.sandbox import load_candidate_from_path

    loaded = load_candidate_from_path(candidate_path, require_forward=False)
    candidate = bind_kernelbench_candidate(
        task,
        loaded.module,
        dtype=torch.float32,
        device="cpu",
    )
    assert torch.equal(candidate(*inputs), first)


def test_kernelbench_parameterized_task_rejects_free_forward_candidate(tmp_path):
    _write_parameterized_model_task(tmp_path)
    task = load_kernelbench_l1_tasks(tmp_path, max_tasks=1)[0]
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text("def forward(x):\n    return x\n", encoding="utf-8")
    from openkernelforge.harness.sandbox import load_candidate_from_path

    loaded = load_candidate_from_path(candidate_path, require_forward=False)
    with pytest.raises(KernelBenchL1Error, match="require candidate class ModelNew"):
        bind_kernelbench_candidate(
            task,
            loaded.module,
            dtype=torch.float32,
            device="cpu",
        )


def test_official_model_task_reuses_one_seeded_init_input_snapshot(tmp_path):
    _write_init_input_model_task(tmp_path)
    task = load_kernelbench_l1_tasks(tmp_path, max_tasks=1)[0]
    task_module = sys.modules[task.metadata["module_name"]]
    assert task.metadata["candidate_contract"] == "model_new"
    assert task.metadata["reference_has_model_state"] is False
    assert task_module.INIT_CALLS == 1

    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text(
        """
from torch import nn

class ModelNew(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale
""",
        encoding="utf-8",
    )
    from openkernelforge.harness.sandbox import load_candidate_from_path

    inputs = task.generate_inputs(7)
    expected = task.reference_fn(*inputs)
    loaded = load_candidate_from_path(candidate_path, require_forward=False)
    candidate = bind_kernelbench_candidate(
        task,
        loaded.module,
        dtype=torch.float32,
        device="cpu",
    )
    assert torch.equal(candidate(*inputs), expected)
    assert task_module.INIT_CALLS == 1

    free_path = tmp_path / "free_candidate.py"
    free_path.write_text("def forward(x):\n    return x\n", encoding="utf-8")
    free = load_candidate_from_path(free_path, require_forward=False)
    with pytest.raises(KernelBenchL1Error, match="require candidate class ModelNew"):
        bind_kernelbench_candidate(task, free.module, dtype=torch.float32, device="cpu")


def test_kernelbench_failed_dynamic_import_does_not_poison_module_cache(tmp_path):
    task_path = _write_synthetic_task(tmp_path, "kb_l1_broken")
    task_path.write_text("raise RuntimeError('broken import')\n", encoding="utf-8")
    with pytest.raises(KernelBenchL1Error):
        load_kernelbench_l1_tasks(tmp_path, max_tasks=1)
    poisoned = [name for name in sys.modules if "kb_l1_broken" in name]
    assert poisoned == []


def test_kernelbench_l1_adapter_fails_clearly_when_dir_missing(tmp_path):
    with pytest.raises(KernelBenchL1Error, match="KernelBench L1 tasks were not found"):
        load_kernelbench_l1_tasks(tmp_path / "missing")


def test_kernelbench_l1_configs_load():
    five = load_config("configs/kernelbench_l1_5task_rigorous.yaml")
    twenty = load_config("configs/kernelbench_l1_20task_rigorous.yaml")
    safe = load_config("configs/kernelbench_l1_20task_rigorous_safe.yaml")
    gemini_five = load_config("configs/kernelbench_l1_5task_gemini_rigorous.yaml")
    gemini_twenty = load_config("configs/kernelbench_l1_20task_gemini_rigorous.yaml")
    gemini_repair = load_config("configs/kernelbench_l1_20task_gemini_repair1.yaml")
    assert five.benchmark.timing_mode == "cuda_event"
    assert five.benchmark.cache_flush.enabled is True
    assert five.benchmark.independent_sessions == 3
    assert twenty.benchmark.repeats == 120
    assert safe.benchmark.repeats == 100
    assert safe.benchmark.torch_compile_mode == "max-autotune"
    assert gemini_five.agent.model == "gemini-3.1-flash-lite"
    assert gemini_five.agent.api_key_env == "GEMINI_API_KEY"
    assert gemini_twenty.benchmark.torch_compile_mode == "max-autotune"
    assert gemini_repair.agent.model == "gemini-3.1-flash-lite"
    assert gemini_repair.execution.disabled_reason
    assert twenty.execution.require_cuda is True


def test_kernelbench_l1_memory_estimator_counts_tensor_bytes():
    x = torch.zeros((2, 3), dtype=torch.float32)
    y = torch.zeros((4,), dtype=torch.int64)
    estimate = estimate_input_memory((x, {"y": y}))
    assert estimate["tensor_count"] == 2
    assert estimate["total_bytes"] == x.numel() * x.element_size() + y.numel() * y.element_size()
    assert estimate["max_tensor_numel"] == 6


def test_kernelbench_task_memory_estimate_accounts_for_verifier_copies():
    x = torch.zeros((2, 3), dtype=torch.float32)

    class Reference:
        state_bytes = 64

    synthetic = type("SyntheticTask", (), {"reference_fn": Reference()})()
    estimate = estimate_task_memory(synthetic, (x,), known_overhead_bytes=32)

    assert estimate["verification_input_copy_factor"] == 5
    assert estimate["model_state_copy_factor"] == 2
    assert estimate["estimated_known_peak_bytes"] == 5 * x.numel() * x.element_size() + 2 * 64 + 32


def test_kernelbench_large_shape_and_memory_preflight_use_meta_tensors(tmp_path):
    _write_metadata_only_large_task(tmp_path)
    task = load_kernelbench_l1_tasks(tmp_path, max_tasks=1)[0]

    inputs = generate_inputs_for_memory_estimate(task)
    estimate = estimate_input_memory(inputs)

    assert task.benchmark_shapes == [(1_000_000_000,)]
    assert inputs[0].device.type == "meta"
    assert estimate["max_tensor_numel"] == 1_000_000_000
    assert estimate["total_bytes"] == 4_000_000_000


def test_kernelbench_memory_preflight_fails_closed_when_meta_is_unsupported(tmp_path):
    task_root = tmp_path / "kb"
    _write_meta_unsupported_task(task_root)
    config_path = tmp_path / "config.yaml"
    _write_oversized_config(config_path, tmp_path / "runs")

    report = run_kernelbench_l1_check(config_path, task_root)
    data = json.loads((report.parent / "kernelbench_l1_check.json").read_text(encoding="utf-8"))

    assert data["status"] == "completed"
    assert data["tasks_skipped"] == 1
    assert data["skipped_reasons"] == {"MEMORY_ESTIMATE_UNAVAILABLE": 1}
    assert data["records"][0]["skip_reason"] == "MEMORY_ESTIMATE_UNAVAILABLE"
    assert "CPU materialization is disabled" in data["records"][0]["error"]


def test_kernelbench_loader_does_not_invent_shape_when_meta_inference_fails(tmp_path):
    _write_meta_unsupported_task(tmp_path)

    task = load_kernelbench_l1_tasks(tmp_path, max_tasks=1)[0]

    assert task.benchmark_shapes == [()]
    assert task.metadata["shape_metadata"]["inference_status"] == (
        "unavailable_without_input_materialization"
    )


def test_kernelbench_loader_does_not_invent_shape_without_shape_metadata(tmp_path):
    task_dir = tmp_path / "level1"
    task_dir.mkdir(parents=True)
    (task_dir / "shape_unknown.py").write_text(
        """
import torch

TASK_ID = "shape_unknown"

def reference_fn(x):
    return x

def input_generator(seed, shape, dtype, device):
    del seed, shape
    return (torch.ones(4, dtype=dtype, device=device),)
""",
        encoding="utf-8",
    )

    task = load_kernelbench_l1_tasks(tmp_path, max_tasks=1)[0]

    assert task.benchmark_shapes == [()]
    assert task.metadata["shape_metadata"]["inference_status"] == (
        "unavailable_without_input_materialization"
    )


def test_kernelbench_l1_check_creates_report_from_synthetic_tasks(tmp_path):
    task_root = tmp_path / "kb"
    _write_synthetic_task(task_root)
    config_path = tmp_path / "config.yaml"
    _write_synthetic_config(config_path, tmp_path / "runs")

    report = run_kernelbench_l1_check(config_path, task_root)

    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "KernelBench L1 Pilot Report" in text
    assert "`kb_l1_add`" in text
    data = json.loads((report.parent / "kernelbench_l1_check.json").read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert data["records"][0]["reference_ok"] is True
    saved_config = (report.parent / "config.yaml").read_text(encoding="utf-8")
    assert "kernelbench:" in saved_config
    assert "max_tasks: 5" in saved_config


def test_kernelbench_l1_check_skips_oversized_synthetic_task(tmp_path):
    task_root = tmp_path / "kb"
    _write_synthetic_task(task_root)
    config_path = tmp_path / "config.yaml"
    _write_oversized_config(config_path, tmp_path / "runs")

    report = run_kernelbench_l1_check(config_path, task_root)

    data = json.loads((report.parent / "kernelbench_l1_check.json").read_text(encoding="utf-8"))
    assert data["tasks_skipped"] == 1
    assert data["skipped_reasons"] == {"ESTIMATED_INPUT_NUMEL_TOO_LARGE": 1}
    assert data["records"][0]["skipped"] is True
    assert data["records"][0]["skip_reason"] == "ESTIMATED_INPUT_NUMEL_TOO_LARGE"
    text = report.read_text(encoding="utf-8")
    assert "Tasks skipped before timing: 1" in text
    assert "ESTIMATED_INPUT_NUMEL_TOO_LARGE" in text
    assert "Benchmark failures: 0" in text


def test_kernelbench_skip_counts_survive_when_skip_records_are_disabled(tmp_path):
    task_root = tmp_path / "kb"
    _write_synthetic_task(task_root)
    config_path = tmp_path / "config.yaml"
    _write_oversized_config(config_path, tmp_path / "runs")
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "record_skipped_tasks: true",
            "record_skipped_tasks: false",
        ),
        encoding="utf-8",
    )

    report = run_kernelbench_l1_check(config_path, task_root)

    data = json.loads((report.parent / "kernelbench_l1_check.json").read_text(encoding="utf-8"))
    assert data["tasks_scanned"] == 1
    assert data["tasks_skipped"] == 1
    assert data["skipped_reasons"] == {"ESTIMATED_INPUT_NUMEL_TOO_LARGE": 1}
    assert data["records"] == []
    assert data["kernelbench_selection"]["pool_scan_complete"] is True


def test_kernelbench_l1_gemini_provider_generates_and_records_candidate(tmp_path, monkeypatch):
    class Backend:
        def generate(self, prompt, *, system=None, **kwargs):
            assert "KernelBench task source" in prompt
            return "```python\nimport torch\n\ndef forward(x, y):\n    return x + y\n```"

    monkeypatch.setattr("openkernelforge.reports.kernelbench_l1.create_backend", lambda agent: Backend())
    task_root = tmp_path / "kb"
    _write_synthetic_task(task_root)
    config_path = tmp_path / "config.yaml"
    _write_synthetic_gemini_config(config_path, tmp_path / "runs")

    report = run_kernelbench_l1_check(config_path, task_root)

    data = json.loads((report.parent / "kernelbench_l1_check.json").read_text(encoding="utf-8"))
    candidates = data["candidate_records"]
    assert len(candidates) == 1
    assert candidates[0]["policy_passed"] is True
    assert candidates[0]["verification_passed"] is True
    assert candidates[0]["benchmarked"] is True
    assert Path(candidates[0]["prompt_path"]).exists()
    assert Path(candidates[0]["response_path"]).exists()
    assert Path(candidates[0]["candidate_path"]).exists()
    assert (report.parent / "results.jsonl").exists()
    text = report.read_text(encoding="utf-8")
    assert "Candidates generated: 1" in text
    assert "Verification passed/failed: 1/0" in text


def test_kernelbench_gemini_provider_rejects_ignored_multi_candidate_budget(tmp_path):
    task_root = tmp_path / "kb"
    _write_synthetic_task(task_root)
    config_path = tmp_path / "config.yaml"
    _write_synthetic_gemini_config(config_path, tmp_path / "runs")
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "candidates_per_task: 1",
            "candidates_per_task: 2",
        ),
        encoding="utf-8",
    )

    report = run_kernelbench_l1_check(config_path, task_root)
    data = json.loads((report.parent / "kernelbench_l1_check.json").read_text(encoding="utf-8"))

    assert data["status"] == "failed"
    assert "exactly one candidate per task" in data["error"]


def test_kernelbench_cloud_provider_fails_before_generation_without_env_key(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    task_root = tmp_path / "kb"
    _write_synthetic_task(task_root)
    config_path = tmp_path / "config.yaml"
    _write_synthetic_gemini_config(config_path, tmp_path / "runs")
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        .replace("backend: fake", "backend: openai_compatible")
        .replace("model: test-gemini", "model: test-gemini\n  base_url: https://example.invalid/v1"),
        encoding="utf-8",
    )

    report = run_kernelbench_l1_check(config_path, task_root)
    data = json.loads((report.parent / "kernelbench_l1_check.json").read_text(encoding="utf-8"))

    assert data["status"] == "failed"
    assert data["error"] == "Missing API key environment variable: export GEMINI_API_KEY=<key>"


def test_kernelbench_gemini_provider_rejects_credentials_in_headers(tmp_path):
    task_root = tmp_path / "kb"
    _write_synthetic_task(task_root)
    config_path = tmp_path / "config.yaml"
    _write_synthetic_gemini_config(config_path, tmp_path / "runs")
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "allow_torch_fallback: true",
            "allow_torch_fallback: true\n  extra_headers:\n    Authorization: Bearer embedded",
        ),
        encoding="utf-8",
    )

    report = run_kernelbench_l1_check(config_path, task_root)
    data = json.loads((report.parent / "kernelbench_l1_check.json").read_text(encoding="utf-8"))

    assert data["status"] == "failed"
    assert "agent.extra_headers.Authorization" in data["error"]
    saved_config = (report.parent / "config.yaml").read_text(encoding="utf-8")
    assert "Bearer embedded" not in saved_config
    assert "<redacted>" in saved_config


def test_kernelbench_backend_failure_preserves_prompt_and_candidate_record(
    tmp_path,
    monkeypatch,
):
    class FailingBackend:
        def generate(self, prompt, *, system=None, **kwargs):
            raise RuntimeError("synthetic backend outage")

    task_root = tmp_path / "kb"
    _write_synthetic_task(task_root)
    config_path = tmp_path / "config.yaml"
    _write_synthetic_gemini_config(config_path, tmp_path / "runs")
    monkeypatch.setattr(
        "openkernelforge.reports.kernelbench_l1.create_backend",
        lambda config: FailingBackend(),
    )

    report = run_kernelbench_l1_check(config_path, task_root)
    data = json.loads((report.parent / "kernelbench_l1_check.json").read_text(encoding="utf-8"))

    assert data["status"] == "completed"
    assert len(data["candidate_records"]) == 1
    candidate = data["candidate_records"][0]
    assert candidate["failure_reason"] == "backend_generation_failed"
    assert candidate["candidate_label"] == "GENERATION_FAILED"
    assert candidate["candidate_path"] is None
    assert candidate["response_path"] is None
    assert Path(candidate["prompt_path"]).exists()
    assert Path(candidate["error_log_path"]).exists()
    assert "synthetic backend outage" in Path(candidate["error_log_path"]).read_text(encoding="utf-8")


def test_kernelbench_existing_file_provider_evaluates_candidate(tmp_path):
    task_root = tmp_path / "kb"
    _write_synthetic_task(task_root)
    candidate_root = tmp_path / "existing"
    candidate_root.mkdir()
    source_path = candidate_root / "kb_l1_add.py"
    source_path.write_text("def forward(x, y):\n    return x + y\n", encoding="utf-8")
    config_path = tmp_path / "existing.yaml"
    _write_existing_file_config(config_path, tmp_path / "runs", candidate_root)

    report = run_kernelbench_l1_check(config_path, task_root)

    data = json.loads((report.parent / "kernelbench_l1_check.json").read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert len(data["candidate_records"]) == 1
    candidate = data["candidate_records"][0]
    assert candidate["source_type"] == "existing_file"
    assert candidate["source_candidate_path"] == str(source_path)
    assert candidate["policy_passed"] is True
    assert candidate["verification_passed"] is True
    assert candidate["benchmarked"] is True
    assert "Candidates evaluated: 1" in report.read_text(encoding="utf-8")


def test_kernelbench_existing_file_provider_fails_when_task_candidate_is_missing(tmp_path):
    task_root = tmp_path / "kb"
    _write_synthetic_task(task_root)
    candidate_root = tmp_path / "empty"
    candidate_root.mkdir()
    config_path = tmp_path / "existing.yaml"
    _write_existing_file_config(config_path, tmp_path / "runs", candidate_root)

    report = run_kernelbench_l1_check(config_path, task_root)

    data = json.loads((report.parent / "kernelbench_l1_check.json").read_text(encoding="utf-8"))
    assert data["status"] == "completed_with_failures"
    assert "No existing candidate file found" in data["records"][0]["error"]


def test_kernelbench_l1_gemini_repair_provider_includes_parent_metadata(tmp_path, monkeypatch):
    class Backend:
        def generate(self, prompt, *, system=None, **kwargs):
            assert "Repair one failed Python candidate" in prompt
            assert "Original failed candidate" in prompt
            assert "shape mismatch" in prompt
            return "```python\nimport torch\n\ndef forward(x, y):\n    return x + y\n```"

    monkeypatch.setattr("openkernelforge.reports.kernelbench_l1.create_backend", lambda agent: Backend())
    task_root = tmp_path / "kb"
    _write_synthetic_task(task_root)
    failed_candidate = tmp_path / "failed_candidate.py"
    failed_candidate.write_text("def forward(x, y):\n    return x\n", encoding="utf-8")
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(
            {
                "selected_for_repair": [
                    {
                        "task_id": "kb_l1_add",
                        "candidate_path": str(failed_candidate),
                        "parent_run_dir": "runs/original",
                        "failure_category": "shape mismatch",
                        "repairability": "high",
                        "suggested_repair_instruction": "Return x + y with the same shape.",
                        "verification_error": "shape mismatch",
                        "parent_policy_version": "ast-v5",
                        "parent_candidate_contract": "forward",
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "repair.yaml"
    _write_synthetic_gemini_repair_config(config_path, tmp_path / "runs", taxonomy_path)

    report = run_kernelbench_l1_check(config_path, task_root)

    data = json.loads((report.parent / "kernelbench_l1_check.json").read_text(encoding="utf-8"))
    candidate = data["candidate_records"][0]
    assert candidate["source_type"] == "gemini_repair"
    assert candidate["generation_stage"] == "kernelbench_l1_gemini_repair1"
    assert candidate["parent_candidate_path"] == str(failed_candidate)
    assert candidate["failure_category"] == "shape mismatch"
    assert candidate["verification_passed"] is True


def test_repair_index_rejects_unversioned_historical_parent(tmp_path):
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(
            {
                "selected_for_repair": [
                    {
                        "task_id": "kb_l1_add",
                        "candidate_path": "runs/historical/candidate.py",
                        "repairability": "high",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KernelBenchL1Error, match="historical or unversioned"):
        _load_repair_index(
            {
                "repair_taxonomy_path": str(taxonomy_path),
                "repair_task_ids": ["kb_l1_add"],
            }
        )


def test_repair_contract_must_match_loaded_task_contract(tmp_path):
    _write_init_input_model_task(tmp_path)
    task = load_kernelbench_l1_tasks(tmp_path, max_tasks=1)[0]

    with pytest.raises(KernelBenchL1Error, match="does not match"):
        _validate_repair_contracts(
            {
                task.task_id: {
                    "task_id": task.task_id,
                    "parent_candidate_contract": "forward",
                }
            },
            [task],
        )


def test_kernelbench_compile_failure_marks_run_incomplete(tmp_path, monkeypatch):
    task_root = tmp_path / "kb"
    _write_synthetic_task(task_root)
    config_path = tmp_path / "compile.yaml"
    _write_synthetic_config(config_path, tmp_path / "runs")
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "include_torch_compile: false",
            "include_torch_compile: true",
        ),
        encoding="utf-8",
    )
    stats = RuntimeStats(1.0, 1.0, 1.0, 1.0, samples_ms=[1.0])

    def failed_compile(*args, **kwargs):
        return BenchmarkResult(
            task_id="kb_l1_add",
            candidate_name="reference_baseline",
            shape=(4, 4),
            dtype="float32",
            device="cpu",
            eager=stats,
            candidate=stats,
            compile_error="compile failed",
        )

    monkeypatch.setattr("openkernelforge.reports.kernelbench_l1.benchmark_task", failed_compile)
    report = run_kernelbench_l1_check(config_path, task_root)

    data = json.loads((report.parent / "kernelbench_l1_check.json").read_text(encoding="utf-8"))
    assert data["status"] == "completed_with_failures"
    assert data["records"][0]["benchmark_summary"]["compile_error"] == "compile failed"


def test_kernelbench_l1_report_generator_handles_no_candidate_results(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    data = {
        "run_dir": str(run_dir),
        "status": "completed",
        "kernelbench_dir": "synthetic",
        "tasks_loaded": 1,
        "timing": {"timing_mode": "wall_clock", "cache_flush_enabled": False, "independent_sessions": 1},
        "environment": {"python_version": "test", "platform": "test", "viability": "CPU_ONLY"},
        "records": [
            {
                "task_id": "kb_l1_add",
                "op_family": "elementwise",
                "shape": [4, 4],
                "reference_ok": True,
                "candidate_path": None,
                "benchmark_summary": {"eager_median_ms": 0.1},
            }
        ],
    }
    report = write_kernelbench_l1_report(run_dir, data=data)
    text = report.read_text(encoding="utf-8")
    assert "Candidate generation is intentionally optional" in text
    assert "Single-run wins: none recorded" in text


def test_kernelbench_l1_report_separates_skipped_memory_tasks(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    data = {
        "run_dir": str(run_dir),
        "status": "completed",
        "kernelbench_dir": "synthetic",
        "tasks_loaded": 2,
        "tasks_selected": 1,
        "tasks_skipped": 1,
        "skipped_reasons": {"ESTIMATED_MEMORY_TOO_LARGE": 1},
        "timing": {"timing_mode": "wall_clock", "cache_flush_enabled": False, "independent_sessions": 1},
        "environment": {"python_version": "test", "platform": "test", "viability": "CPU_ONLY"},
        "records": [
            {
                "task_id": "too_big",
                "op_family": "activation",
                "shape": [4096, 393216],
                "reference_ok": False,
                "candidate_path": None,
                "skipped": True,
                "skip_reason": "ESTIMATED_MEMORY_TOO_LARGE",
                "benchmark_summary": None,
            },
            {
                "task_id": "kb_l1_add",
                "op_family": "elementwise",
                "shape": [4, 4],
                "reference_ok": True,
                "candidate_path": None,
                "skipped": False,
                "benchmark_summary": {"eager_median_ms": 0.1},
            },
        ],
    }
    report = write_kernelbench_l1_report(run_dir, data=data)
    text = report.read_text(encoding="utf-8")
    assert "Tasks skipped before timing: 1" in text
    assert "`ESTIMATED_MEMORY_TOO_LARGE`: 1" in text
    assert "| `too_big` | activation | `[4096, 393216]` | n/a | n/a | skipped | ESTIMATED_MEMORY_TOO_LARGE |" in text


def test_kernelbench_candidate_provider_llm_later_raises_clear_error():
    provider = make_candidate_provider({"candidate_provider": "llm_later"})
    with pytest.raises(NotImplementedError, match="not implemented"):
        provider.candidate_for_task("task")


def test_kernelbench_l1_cli_check_creates_report(tmp_path):
    task_root = tmp_path / "kb"
    _write_synthetic_task(task_root)
    config_path = tmp_path / "config.yaml"
    _write_synthetic_config(config_path, tmp_path / "runs")

    code = main(
        [
            "kernelbench-l1-check",
            "--config",
            str(config_path),
            "--kernelbench-dir",
            str(task_root),
        ]
    )

    assert code == 0
    reports = list((tmp_path / "runs").glob("*/kernelbench_l1_check.md"))
    assert reports
