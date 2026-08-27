# Model Identifier Audit

This audit records model identifiers exactly as configured in the preserved
configs and run records. It does not infer provider-side model revisions.

| Source | Configured model string | Response model field | Backend | Artifact source | Verification status |
| --- | --- | --- | --- | --- | --- |
| fused8 template | not applicable | not applicable | fake/template | `configs/template_fused8_gpu_benchmark_rigorous.yaml`; summarized run `20260520_155839` | 160/160 verified |
| fused8 Gemini | `gemini-3.1-flash-lite` | not preserved | OpenAI-compatible Gemini endpoint | `configs/gemini_fused8_gpu_baseline_rigorous.yaml`; summarized run `20260520_163344` | 23/24 verified |
| fused8 OpenAI mini | `gpt-5.4-mini` | not preserved | OpenAI-compatible OpenAI chat completions | `configs/openai_mini_fused8_gpu_baseline_rigorous.yaml`; summarized run `20260520_163607` | 12/24 verified |
| KernelBench one-shot Gemini | `gemini-3.1-flash-lite` | not preserved; JSONL records preserve configured `model` field | OpenAI-compatible Gemini endpoint | `artifacts/runpod_imports/runs/20260520_202314/config.yaml` and `results.jsonl` | historical evaluator recorded 3/20; invalid as correctness evidence |
| KernelBench repair Gemini | `gemini-3.1-flash-lite` | not preserved; JSONL records preserve configured `model` field | OpenAI-compatible Gemini endpoint | `artifacts/runpod_imports/runs/20260520_213128/config.yaml` and `results.jsonl` | historical evaluator recorded 1/8; invalid as repair evidence |

Paper wording should therefore describe these identifiers as configured API
strings. The local artifact package does not preserve provider response
`model` fields for the fused8 runs. KernelBench JSONL records preserve the
configured `model` field for every candidate record, but not a separate
provider-returned model-version field. The KernelBench verification counts in
this audit are provenance fields from the affected adapter, not supported
correctness results.
