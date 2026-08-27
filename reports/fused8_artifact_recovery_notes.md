# Fused8 Artifact Recovery Notes

This recovery pass searches the local workspace and `artifacts/runpod_imports` for the rigorous fused8 run directories. It does not run benchmarks or infer missing interval statistics.

## Recovery Status

| Source | Run | Artifact path | IQR | Bootstrap CI | Std/CV | Per-session medians | Flip pairs | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| template | 20260520_155839 | `missing` | not preserved | not preserved | not preserved | not preserved | not preserved | full run directory is missing locally and under artifacts/runpod_imports |
| Gemini | 20260520_163344 | `missing` | not preserved | not preserved | partially preserved | not preserved | not preserved | full run directory is missing locally and under artifacts/runpod_imports |
| OpenAI mini | 20260520_163607 | `missing` | not preserved | not preserved | partially preserved | not preserved | not preserved | full run directory is missing locally and under artifacts/runpod_imports |

## Flip-Frequency Status

- fused8 template task-best summary: `0.2500` (task-best imported summary only; bias_relu is the observed flip)
- fused8 template all 160 candidates: `not preserved` (full per-candidate single-run and repeat-median pairs are not preserved locally)

The local package therefore retains the task-best `bias_relu` flip but not the full 160-candidate single-run-to-repeat flip frequency. If the missing RunPod fused8 directories are later imported, rerun `python scripts/analyze_existing_results_statistics.py` and rebuild paper assets.
