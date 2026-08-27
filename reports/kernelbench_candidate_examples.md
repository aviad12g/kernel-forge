# KernelBench Candidate Examples

This note records the concrete examples used in the paper appendix. It uses
existing imported artifacts only and does not add new benchmark results. Both
examples come from the affected historical evaluator and are retained for
auditability, not as corrected-adapter correctness or performance evidence.

## Historically Accepted Candidate: CrossEntropyLoss

- Run: `artifacts/runpod_imports/runs/20260520_202314`
- Candidate: `candidates/KernelBench__level1__95_CrossEntropyLoss/candidate_000.py`
- Historical label: `REPEAT_STABLE_WIN`
- Historically recorded speedup: 1.992x vs eager, 2.895x vs compile
- Pattern: row-wise log-sum-exp, target-logit gather, mean reduction.
- Current status: the candidate fails `ast-v5` because it uses a Torch mean
  fallback, and the old reference lifecycle invalidates speed attribution.
- Interpretation: the source is useful for understanding what the old
  evaluator accepted. No corrected correctness, speedup, or profiler
  attribution is claimed.

## Failed Candidate: 4D Tensor Matmul

- Run: `artifacts/runpod_imports/runs/20260520_202314`
- Candidate: `candidates/KernelBench__level1__11_4D_tensor_matrix_multiplication/candidate_000.py`
- Historical policy: passed
- Verification: failed before timing
- Failure category: `Triton compile error`
- Verifier summary: `Cannot broadcast, rank mismatch: ['1','1','32'], ['16','1','1','32']`
- Repairability: high
- Repair result: failed in the single repair pass
- Interpretation: failure is localized to Triton pointer arithmetic and
  load-mask shape rather than broad task selection or benchmarking.
