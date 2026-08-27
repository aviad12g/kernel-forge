# KernelBench Loss Profiler Summary

This diagnostic is intentionally separate from CUDA-event benchmark results. It does not create benchmark speedups, repeatability labels, or generated candidates. Historical adapter outputs are blocked from execution by default and are never paper evidence.

- Profiler skipped: CUDA is not available in this workspace.

## Candidate Artifact Availability

| Task | Candidate source | Metadata | Historical | Contract metadata | Current policy |
| --- | --- | --- | --- | --- | --- |
| CrossEntropyLoss | `/Users/mazalcohen/Downloads/kernel try/artifacts/runpod_imports/runs/20260520_202314/candidates/KernelBench__level1__95_CrossEntropyLoss/candidate_000.py` (present) | `/Users/mazalcohen/Downloads/kernel try/artifacts/runpod_imports/runs/20260520_202314/results.jsonl` | True | not available / not available | obvious_torch_fallback:loss.mean |
| TripletMarginLoss | `/Users/mazalcohen/Downloads/kernel try/artifacts/runpod_imports/runs/20260520_202314/candidates/KernelBench__level1__99_TripletMarginLoss/candidate_000.py` (present) | `/Users/mazalcohen/Downloads/kernel try/artifacts/runpod_imports/runs/20260520_202314/results.jsonl` | True | not available / not available | obvious_torch_fallback:out.mean |
| KLDivLoss | `/Users/mazalcohen/Downloads/kernel try/artifacts/runpod_imports/runs/20260520_213128/candidates/KernelBench__level1__98_KLDivLoss/candidate_000.py` (present) | `/Users/mazalcohen/Downloads/kernel try/artifacts/runpod_imports/runs/20260520_213128/results.jsonl` | True | not available / not available | obvious_torch_fallback:torch.log |

## Mechanism Summary

| Task | Existing speedup | Candidate pattern | Attribution confidence | Caveat |
| --- | --- | --- | --- | --- |
| CrossEntropyLoss | 1.992x vs eager, 2.895x vs compile | row-wise log-sum-exp, target gather, per-row loss buffer, final mean reduction | historical source only | historical source is excluded from attribution; current policy result: obvious_torch_fallback:loss.mean |
| TripletMarginLoss | 4.176x vs eager, 3.208x vs compile | single Triton kernel computes positive/negative distances, sqrt, hinge, and per-row loss before mean | historical source only | historical source is excluded from attribution; current policy result: obvious_torch_fallback:out.mean |
| KLDivLoss | 1.843x vs eager, 1.028x vs compile | torch.log on predictions plus Triton KL elementwise term and torch.sum batchmean reduction | historical source only | historical source is excluded from attribution; current policy result: obvious_torch_fallback:torch.log |

## Profiler Rows

- Profiled operator rows: 0
- Output CSV: `reports/tables/kernelbench_loss_profiler_ops.csv`
- Memory CSV: `reports/tables/kernelbench_loss_profiler_memory.csv`
- Mechanism CSV: `reports/tables/kernelbench_loss_mechanism_summary.csv`

Only corrected-run profiler rows may support mechanism discussion. Historical debug rows remain excluded because candidate selection and baseline timing used the affected adapter.
