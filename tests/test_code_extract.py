from openkernelforge.agents.code_extract import extract_python_code


def test_extract_python_code_from_markdown_fence():
    response = """
Here is the candidate:

```python
import torch


def forward(x):
    return torch.relu(x)
```

That should pass.
"""
    result = extract_python_code(response)
    assert result.ok
    assert result.code is not None
    assert "def forward" in result.code
    assert result.metadata["source"] == "fenced"


def test_extract_python_code_from_raw_response():
    response = "import torch\n\n\ndef forward(x, y):\n    return x + y\n"
    result = extract_python_code(response)
    assert result.ok
    assert result.code == response


def test_extract_python_code_reports_error_without_forward():
    result = extract_python_code("This response has no code.")
    assert not result.ok
    assert result.error is not None
