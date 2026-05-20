# Experiments

## Three-Task Sandbox

The initial sandbox used `vector_add`, `relu`, and `bias_relu` to validate correctness, benchmarking, deterministic templates, and repeatability. It showed that simple standalone elementwise tasks were poor performance targets. Apparent single-run wins did not reliably survive repeatability checks.

| Task | Best single-run speedup | Repeat outcome | Conclusion |
| --- | ---: | --- | --- |
| `vector_add` | 0.692x | 0.483x repeat median | poor standalone target |
| `relu` | 0.812x | 0.512x repeat median | poor standalone target |
| `bias_relu` | 1.017x | 0.705x repeat median | single-run win was unstable |

## Internal Fused8 Benchmark

The fused8 benchmark uses:

- `bias_relu`
- `sigmoid_mul`
- `add_relu`
- `residual_add_relu`
- `bias_gelu`
- `row_sum`
- `layernorm_small`
- `rmsnorm_small`

These workloads are more appropriate for Triton than isolated elementwise kernels because fusion and reduction structure can amortize launch overhead.

## Deterministic Template Results

The deterministic fused8 template wide run generated 2076 candidates and verified 2076/2076. Single-run bests beat eager on 5/8 tasks, but repeatability confirmed stable above-eager wins on 3 tasks:

- `residual_add_relu`: 1.168x repeat median
- `bias_gelu`: 1.657x repeat median
- `rmsnorm_small`: 1.802x repeat median

## Model Results

| Baseline | Candidates | Verified | Median speedup vs eager | Repeat-stable wins | Interpretation |
| --- | ---: | --- | ---: | --- | --- |
| Gemini baseline | 28 | 28/28 | 0.933x | competitive but not dominant | correct fused kernels reliably |
| Gemini template-guided | 34 | 34/34 | 0.798x | `residual_add_relu` | useful data, worse median |
| OpenAI mini cheap | 8 | 8/8 | 0.882x | `residual_add_relu`, `bias_gelu`, `rmsnorm_small` | cheap and competitive |
| GPT-5.5 cheap | 8 | 8/8 | 0.927x | `bias_gelu`, `rmsnorm_small` | not clearly better under cheap budget |
| Qwen 7B local | 8 | 1/8 effective | 0.002x | none | weak zero-shot |

Qwen 14B was not evaluated because vLLM failed with disk/cache capacity exhaustion during model download. It should not be interpreted as a model-quality failure.

## Interpretation

The central experimental result is that correctness became reliable before speed did. Correct candidates were common for stronger API models, but repeat-stable speedups were rarer. Deterministic templates remain the clearest baseline for future comparisons.
