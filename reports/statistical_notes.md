# Statistical Notes From Existing Artifacts

This analysis uses only existing CSV and run artifacts. It does not run experiments, call model APIs, or relabel results. KernelBench rows come from the affected historical adapter and are audit metadata, not model-accuracy or performance evidence.

## Verification-Rate Intervals

Verification rates use 95% Wilson intervals under the fixed candidate budgets.

| Study | Source | Successes | Trials | Rate | Wilson 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| fused8 | template | 160 | 160 | 1.0000 | [0.9766, 1.0000] |
| fused8 | Gemini | 23 | 24 | 0.9583 | [0.7976, 0.9926] |
| fused8 | OpenAI mini | 12 | 24 | 0.5000 | [0.3143, 0.6857] |
| Historical KB (affected): pilot | Gemini one-shot | 3 | 20 | 0.1500 | [0.0524, 0.3604] |
| Historical KB (affected): repair1 | Gemini repair | 1 | 8 | 0.1250 | [0.0224, 0.4709] |
| Historical KB (affected): combined | Gemini one-shot + repair1 | 4 | 20 | 0.2000 | [0.0807, 0.4160] |

Candidate-level model significance testing is omitted because multiple candidates share each fused8 task and are not independent Bernoulli trials.

## KernelBench Family Summary

The affected historical evaluator recorded nonuniform outcomes by family. The table documents its behavior; it is not a corrected-adapter family estimate.

| Family | Selected | One-shot verified | One-shot stable | Repair attempted | Repair verified | Combined correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| convolution | 7 | 0 | 0 | 0 | 0 | 0 |
| matmul | 7 | 1 | 0 | 6 | 0 | 1 |
| pooling | 3 | 0 | 0 | 1 | 0 | 0 |
| loss | 3 | 2 | 2 | 1 | 1 | 3 |

## Single-Run Versus Repeat Flips

- fused8 template task-best summary: single-run above eager = 4, repeat below eager = 1, flip rate = 0.2500. task-best imported summary only; bias_relu is the observed flip
- fused8 template all 160 candidates: single-run above eager = not preserved, repeat below eager = not preserved, flip rate = not preserved. full per-candidate single-run and repeat-median pairs are not preserved locally

## Memory Filtering

Historical KernelBench memory filtering is characterized in `reports/tables/kernelbench_memory_filter_summary.csv`. It was a lower-bound, deterministic selector and is retained as audit provenance.

## Caveats

- Fused8 stable-win counts are small and should not be read as statistically significant rankings.
- KernelBench verification, family, and stable-label counts are provisional because the source adapter was invalid.
- Candidate search creates multiplicity; repeatability labels reduce but do not eliminate selection bias.
- Three independent sessions are a practical stability check, not a high-powered variance estimate.
