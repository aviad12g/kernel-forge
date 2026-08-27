# Prompt Templates And Decoding Settings

This document summarizes the prompt and decoding settings used by the paper-facing fused8 and KernelBench runs. Full prompts, responses, and extracted candidates are preserved in run artifacts when the corresponding run directory is available.

## Fused8 Candidate Prompt

Fused8 model runs use `openkernelforge.agents.prompt_templates.build_task_prompt` with `prompt_version = v2_task_skeletons`.

Representative structure:

```text
Write a complete Python candidate file for OpenKernelForge.
Prompt version: v2_task_skeletons
The file must expose def forward(*args): ... at module scope.
Allowed imports: torch, triton, and triton.language as tl.
Do not fake outputs, cache expected tensors, read hidden files, or call the reference function.
Prefer a simple correct Triton kernel over a complex broken kernel.
For elementwise tasks, use a straightforward block-based Triton kernel.
Include all required imports.
Torch fallback mode is disabled: do not use plain PyTorch fallback code ...
Return only Python code, with no explanation outside code.

Task id: <task_id>
Name: <task_name>
Description: <task_description>
Allowed dtypes: float32
Correctness tolerance: rtol=<rtol>, atol=<atol>
Benchmark shapes: <shape>
Task-specific hint: <hint>
Additional task hints:
- <task-specific implementation hints>
Task-specific Triton skeleton hint:
- <optional skeleton hint>
Reference implementation:
<reference source>
```

The system prompt for `LLMAgent` is:

```text
You generate concise Python candidate kernels for local verification. Return code that can be imported as a Python module.
```

Extraction uses `extract_python_code`, which prefers fenced code blocks but also accepts raw Python if it parses and defines module-level `forward`.

## Fused8 Model Settings

| Source | Model | Backend | Candidates | Temperature | top_p | Token limit |
| --- | --- | --- | --- | --- | --- | --- |
| Gemini | `gemini-3.1-flash-lite` | OpenAI-compatible Gemini endpoint | 3 per task | 0.2 | 0.95 | 4096 max tokens |
| OpenAI mini | `gpt-5.4-mini` | OpenAI-compatible `/v1/chat/completions` | 3 per task | null | null | `extra_body.max_completion_tokens = 2048` |

Both runs use `max_attempts = 1`, `stop_after_first_correct = false`, `benchmark_all_correct = true`, `allow_torch_fallback = false`, and no performance search.

These model identifiers are reported as configured API strings. The local
artifact package does not preserve provider response `model` fields for the
fused8 runs, so the paper does not infer provider-side model revisions.

## KernelBench One-Shot Prompt

KernelBench generation uses `candidate_provider = gemini` in `openkernelforge/reports/kernelbench_l1.py`.

Representative structure:

````text
Write one Python candidate for a KernelBench L1 task.
Return only a fenced Python code block.
For every official KernelBench task that defines Model, define:
class ModelNew(torch.nn.Module):
    def __init__(self, <same arguments as Model>): ...
    def forward(self, *args): ...

Contract:
- ModelNew receives get_init_inputs() and forward receives get_inputs().
- It must match Model.forward for both inputs and initialized model state.
- Use Triton kernels when appropriate.
- Do not call the PyTorch reference, Model class, get_inputs, or any KernelBench/OpenKernelForge task module inside forward.
- Do not fake outputs, cache expected tensors, read files, or use hidden state.
- Torch fallback is disabled: avoid direct torch operations such as torch.matmul, torch.relu, torch.sum, torch.nn.functional.* as the main computation.
- Torch may be used for allocation and wrappers.

Task id: <task_id>
Task name: <task_name>
Op family: <op_family>
Benchmark shape metadata: <shape_metadata>
Benchmark shape: <shape>
Tolerance: rtol=<rtol>, atol=<atol>
KernelBench source path: <source_path>

KernelBench task source:
```python
<task source, truncated in the middle if needed>
```
````

The system prompt is:

```text
You generate concise Python candidate kernels for local verification. Return only a Python code block.
```

Settings: Gemini `gemini-3.1-flash-lite`, `temperature = 0.2`, `top_p = 0.95`, `max_tokens = 4096`, one candidate per task.

KernelBench JSONL records preserve the configured `model` field
`gemini-3.1-flash-lite` for candidate records, but not a separate
provider-returned model-version field.

## KernelBench Repair Prompt

Repair uses `candidate_provider = gemini_repair`. The prompt includes the original task source, original failed candidate, verification error, traceback/failure details, failure category, and suggested repair instruction.

The corrected repair objective explicitly says:

```text
Fix correctness first. Performance is secondary.
Preserve the declared ModelNew or local synthetic forward contract and output.
Do not call the PyTorch reference, Model class, get_inputs, or any KernelBench/OpenKernelForge task module inside forward.
Do not use torch operations as the main computation. Torch is allowed only for allocation/wrapping, tensor shape/device/dtype inspection, and launching Triton kernels.
```

Settings: Gemini `gemini-3.1-flash-lite`, `temperature = 0.2`, `top_p = 0.95`, `max_tokens = 4096`, one repair candidate for each selected repair task.

The preserved 20260520 KernelBench prompts used the older free-function-only
contract. They are retained verbatim as historical artifacts. New prompts use
`ModelNew` for every official task that defines `Model`; no historical response
is rewritten. Free functions remain supported only by the local synthetic
`reference_fn` interface used in CPU-only adapter tests.
