# Appendix

This appendix keeps task-level details out of the main body. Raw prompts, responses, candidate files, verification traces, and timing records remain in the run artifacts.

## Fused8 Task-Level Summary

| Task | Winner | Source | Repeat median | Uncertainty | Interpretation |
| --- | --- | --- | ---: | --- | --- |
| `bias_relu` | none | n/a | 0.976x | not preserved | single-run win fell below eager |
| `sigmoid` | none | n/a | 0.997x | std 0.029, CV 0.030 | below eager |
| `add_relu` | none | n/a | 0.968x | std 0.003, CV 0.003 | below eager |
| `residual` | OpenAI mini | model | 1.074x | std 0.048, CV 0.045 | model-over-template stable win |
| `bias_gelu` | template | template | 1.485x | not preserved | template stronger |
| `row_sum` | none | n/a | 0.674x | not preserved | below eager |
| `layernorm` | none | n/a | 0.791x | not preserved | below eager |
| `rmsnorm` | template | template | 1.452x | not preserved | template strongest |

## Fused8 Statistical Summary

Verification rates use 95% Wilson intervals under fixed candidate budgets. They quantify uncertainty for this protocol and are not model-family rankings.

| Source | Verified | Rate | Wilson 95% CI |
| --- | ---: | ---: | --- |
| template | 160/160 | 1.0000 | [0.9766, 1.0000] |
| Gemini | 23/24 | 0.9583 | [0.7976, 0.9926] |
| OpenAI mini | 12/24 | 0.5000 | [0.3143, 0.6857] |

The candidate-level verification counts are clustered within eight fused8 tasks and are not used for an independent-sample model comparison. The task-best template summary contains four rows above eager in the original summary and one repeat-below-eager flip. The full 160-candidate flip frequency is not preserved locally.

## Fused8 Artifact Recovery

| Source | Run | IQR | Bootstrap CI | Per-candidate flip pairs |
| --- | --- | --- | --- | --- |
| template | 20260520_155839 | not preserved | not preserved | not preserved |
| Gemini | 20260520_163344 | not preserved | not preserved | not preserved |
| OpenAI mini | 20260520_163607 | not preserved | not preserved | not preserved |

The local package preserves fused8 task-best medians and labels, plus a few std/CV fields. Full p25/p75, bootstrap intervals, per-session medians, and all-candidate single-run/repeat pairs require the missing rigorous fused8 run directories.

## Historical KernelBench Adapter Audit

The preserved KernelBench artifacts use the official repository at commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`. They document a 100-task loaded pool, deterministic feasible-subset selection, 20 selected tasks, one generated candidate per selected task, and one repair pass. A post-hoc implementation audit found that these rows cannot support correctness or performance claims:

- the historical prompt forced candidates into a free `forward(*args)` contract and did not expose state created by `get_init_inputs()`;
- every official task that defines `Model` requires a persistent candidate `ModelNew`, including tasks with an empty `state_dict()`;
- the historical reference wrapper reconstructed and transferred `Model` inside each call;
- the old policy accepted high-level Torch fallbacks and import-time model construction that the current policy rejects;
- the input-footprint memory filter was a lower bound, not a complete peak-memory estimate.

The affected evaluator recorded 3/20 one-shot verifications and 1/8 repair verifications. These counts, their Wilson intervals, family breakdowns, speedups, profiler rows, and clock-recorded reruns are historical audit metadata. They are not estimates of Gemini accuracy, repair effectiveness, or KernelBench performance.

## Current Static Policy Re-Audit

The preserved source files were parsed, but not imported or executed, under the current `ast-v5` policy:

| Historical run | Preserved candidates | Current strict-policy pass | Entry-point contract |
| --- | ---: | ---: | --- |
| one-shot `20260520_202314` | 20 | 11/20 | 20/20 free functions |
| repair `20260520_213128` | 8 | 7/8 | 8/8 free functions |

Rejections include high-level Torch convolution/reduction calls and import-time model construction. Passing this static check does not make a free function valid for an official `Model` task, and the AST policy is not a security sandbox. Full rows are in `reports/tables/kernelbench_historical_policy_reaudit.csv`.

## Example Historical Candidates

### Cross-entropy source accepted by the old policy

The preserved source uses a Triton row-wise log-sum-exp and target gather, followed by a high-level Torch mean. The current strict policy rejects the final reduction as `obvious_torch_fallback:loss.mean`. Its historical correctness and speed label are not carried into the supported results.

```python
@triton.jit
def _cross_entropy_kernel(logits_ptr, targets_ptr, loss_ptr,
                          n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    # row-wise log-sum-exp and target gather
    ...
    tl.store(loss_ptr + row_idx, log_sum_exp - target_logit)

def forward(predictions, targets):
    loss = torch.empty(predictions.shape[0], device=predictions.device)
    _cross_entropy_kernel[(predictions.shape[0],)](...)
    return loss.mean()
```

### Four-dimensional matmul compile failure

The preserved `matmul_4d` source failed before timing with `triton.compiler.errors.CompilationError`: pointer and mask ranks did not broadcast. The historical repair selector called this high repairability because the failure was localized to pointer arithmetic; its one repair attempt still failed verification.

```python
a = tl.load(
    A_ptr_base + (rk[None, None, :] + k) * stride_al,
    mask=rk[None, None, :] + k < K,
    other=0.0,
)
```

## Corrected Contract And Required Validation

Current code materializes `get_init_inputs()` once under seed 0, constructs reference `Model` and candidate `ModelNew` instances from that snapshot, moves them to the target device before verification and timing, and rejects free functions for every official `Model` task. It also rotates eager, candidate, and compile order, materializes compilation before runtime sampling, reports session-level speedups, and uses stricter AST policy checks.

No corrected KernelBench CUDA run is included in this package. A corrected baseline and candidate run must complete before KernelBench correctness, family, repair, profiler, clock, or speed claims are restored.
