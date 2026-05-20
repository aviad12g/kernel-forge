import torch

from openkernelforge.tasks.simple_tasks import get_builtin_tasks


def test_builtin_tasks_generate_inputs_and_reference_outputs():
    tasks = get_builtin_tasks()
    assert len(tasks) >= 8

    for task in tasks:
        inputs_a = task.generate_inputs(0, device="cpu")
        inputs_b = task.generate_inputs(0, device="cpu")
        assert inputs_a
        assert len(inputs_a) == len(inputs_b)

        for left, right in zip(inputs_a, inputs_b, strict=True):
            assert isinstance(left, torch.Tensor)
            assert torch.equal(left, right)

        output = task.reference_fn(*inputs_a)
        assert isinstance(output, torch.Tensor)
        assert torch.isfinite(output).all()
