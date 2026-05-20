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

The legacy deterministic fused8 template wide run generated 2076 candidates and verified 2076/2076. Single-run bests beat eager on 5/8 tasks, but repeatability confirmed stable above-eager wins on 3 tasks:

- `residual_add_relu`: 1.168x repeat median
- `bias_gelu`: 1.657x repeat median
- `rmsnorm_small`: 1.802x repeat median

The current paper-facing deterministic template table uses the rigorous CUDA-event run `runs/20260520_155839`. That run used a capped 160-candidate grid with CUDA events, cache flushing, bootstrap intervals, and three independent sessions. It verified 160/160 candidates, had median speedup 0.945x vs eager, and repeated stable above-eager wins on:

- `residual_add_relu`: 1.023x repeat median
- `bias_gelu`: 1.485x repeat median
- `rmsnorm_small`: 1.452x repeat median

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

## Rigorous Timing Status

The model results above remain legacy timing unless explicitly rerun with the rigorous path. The deterministic template results have a new rigorous CUDA-event run. The CUDA-event methodology is implemented, including optional cache flushing, independent sessions, median/IQR/CV summaries, and bootstrap intervals. It was sanity-checked on a RunPod RTX 5090:

```bash
python -m openkernelforge.cli benchmark-methodology-check \
  --config configs/template_fused8_gpu_benchmark_rigorous.yaml \
  --max-tasks 2
```

The methodology check completed at `runs/20260520_145721` with `timing_mode=cuda_event`, cache flushing enabled and performed, and three independent sessions. A small fused8 validation run completed at `runs/20260520_145741` with 160/160 verified candidates and fused8/repeatability reports. The full configured rigorous run completed at `runs/20260520_155839`.

The old 2076-candidate deterministic template table should now be treated as legacy timing. The LLM/OpenAI/Gemini model rows are still legacy timing until those model runs are rerun rigorously.
