import torch

from openkernelforge.harness.inputs import clone_inputs, find_value_difference


def test_clone_inputs_recursively_clones_nested_tensors():
    source = torch.tensor([1.0])
    inputs = ({"nested": [source]},)
    cloned = clone_inputs(inputs)

    cloned[0]["nested"][0].add_(1.0)

    assert source.item() == 1.0
    assert cloned[0]["nested"][0].item() == 2.0


def test_find_value_difference_reports_nested_tensor_path():
    before = ({"nested": [torch.tensor([1.0])]},)
    after = clone_inputs(before)
    after[0]["nested"][0].add_(1.0)

    assert find_value_difference(before, after) == "inputs[0].nested[0]"


def test_find_value_difference_treats_matching_nan_values_as_unchanged():
    before = (torch.tensor([float("nan")]),)
    after = clone_inputs(before)

    assert find_value_difference(before, after) is None
