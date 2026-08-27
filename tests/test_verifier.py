from pathlib import Path

import torch

from openkernelforge.harness.sandbox import load_candidate_from_path
from openkernelforge.harness.verifier import verify_candidate
from openkernelforge.tasks.base import KernelTask, TaskTolerance
from openkernelforge.tasks.simple_tasks import get_task


def _load_tmp_candidate(tmp_path: Path, source: str):
    path = tmp_path / "candidate.py"
    path.write_text(source, encoding="utf-8")
    return load_candidate_from_path(path)


def _task(reference_fn) -> KernelTask:
    def generate(seed, shape, dtype, device):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return (torch.randn(shape, generator=generator, dtype=dtype).to(device),)

    return KernelTask(
        task_id="contract",
        name="contract",
        description="verification contract fixture",
        reference_fn=reference_fn,
        input_generator=generate,
        allowed_dtypes=(torch.float32,),
        tolerance=TaskTolerance(rtol=0.0, atol=0.0),
        benchmark_shapes=[(8,)],
    )


def test_verifier_passes_on_correct_torch_candidate(tmp_path):
    task = get_task("vector_add")
    candidate = _load_tmp_candidate(
        tmp_path,
        "def forward(x, y):\n    return x + y\n",
    )
    result = verify_candidate(
        task,
        candidate.forward,
        seeds=[0, 1],
        shapes=[(16,)],
        device="cpu",
        dtype="float32",
    )
    assert result.passed
    assert all(case.passed for case in result.cases)


def test_verifier_fails_on_wrong_candidate(tmp_path):
    task = get_task("vector_add")
    candidate = _load_tmp_candidate(
        tmp_path,
        "def forward(x, y):\n    return x - y\n",
    )
    result = verify_candidate(
        task,
        candidate.forward,
        seeds=[0],
        shapes=[(16,)],
        device="cpu",
        dtype="float32",
    )
    assert not result.passed
    assert result.cases[0].error_type == "values_not_close"


def test_verifier_rejects_candidate_input_mutation(tmp_path):
    task = get_task("vector_add")
    candidate = _load_tmp_candidate(
        tmp_path,
        "def forward(x, y):\n    x.add_(y)\n    return x\n",
    )
    result = verify_candidate(
        task,
        candidate.forward,
        seeds=[0],
        shapes=[(16,)],
        device="cpu",
        dtype="float32",
    )

    assert not result.passed
    assert result.cases[0].error_type == "unexpected_input_mutation"
    assert "inputs[0]" in (result.cases[0].message or "")


def _special_value_task() -> KernelTask:
    def generate(seed, shape, dtype, device):
        del seed, shape
        return (torch.tensor([0.0, 1.0, -1.0], dtype=dtype, device=device),)

    return KernelTask(
        task_id="special_values",
        name="special values",
        description="NaN and infinity mask parity",
        reference_fn=lambda x: torch.tensor(
            [float("nan"), float("inf"), float("-inf")],
            dtype=x.dtype,
            device=x.device,
        ),
        input_generator=generate,
        allowed_dtypes=(torch.float32,),
        tolerance=TaskTolerance(rtol=0.0, atol=0.0),
        benchmark_shapes=[(3,)],
    )


def test_verifier_accepts_matching_special_value_masks() -> None:
    task = _special_value_task()
    result = verify_candidate(task, task.reference_fn, seeds=[0], device="cpu")
    assert result.passed


def test_verifier_rejects_output_tree_mismatch() -> None:
    task = _task(lambda x: (x + 1, x + 2))

    def candidate(x):
        return [x + 1, x + 2]

    result = verify_candidate(task, candidate, seeds=[0], device="cpu")
    assert not result.passed
    assert result.cases[0].error_type == "output_tree_mismatch"


def test_verifier_rejects_same_input_nondeterminism() -> None:
    task = _task(lambda x: x)
    calls = 0

    def candidate(x):
        nonlocal calls
        calls += 1
        return x + calls

    result = verify_candidate(
        task,
        candidate,
        seeds=[0],
        device="cpu",
        deterministic_repeats=2,
    )
    assert not result.passed
    assert result.cases[0].error_type == "nondeterministic_output_values"


def test_verifier_rejects_alias_contract_mismatch() -> None:
    task = _task(lambda x: x)
    result = verify_candidate(
        task,
        lambda x: x.clone(),
        seeds=[0],
        device="cpu",
        require_alias_contract=True,
    )
    assert not result.passed
    assert result.cases[0].error_type == "alias_contract_mismatch"


def test_verifier_rejects_special_value_mask_mismatch() -> None:
    task = _special_value_task()
    result = verify_candidate(
        task,
        lambda x: torch.tensor(
            [float("nan"), float("-inf"), float("inf")],
            dtype=x.dtype,
            device=x.device,
        ),
        seeds=[0],
        device="cpu",
    )
    assert not result.passed
    assert result.cases[0].error_type == "special_value_mask_mismatch"
