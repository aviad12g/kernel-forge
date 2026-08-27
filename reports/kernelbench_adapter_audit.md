# Historical KernelBench Adapter Audit

## Evidence status

The preserved KernelBench runs are retained for evaluator auditing, not as supported correctness or performance evidence. The historical adapter required a free `forward(*args)` candidate receiving only `get_inputs()`, although every official task that defines `Model` requires a `ModelNew` instance initialized through `get_init_inputs()`. Parameterized tasks additionally had no route to the reference state. The adapter also reconstructed and transferred the reference `Model` inside each reference call. These defects invalidate model-accuracy and speedup interpretations of the affected rows.

The corrected adapter materializes one seeded initialization snapshot and uses persistent reference `Model` and candidate `ModelNew` instances outside verification and timing loops. Every official `Model` task rejects free-function candidates. The completed corrected workshop campaign uses this lifecycle and is reported separately in `reports/workshop2026_corrected_results.md`; the historical rows audited here remain excluded.

## Static policy re-audit

This re-audit parses preserved source with the current `ast-v5` policy. It does not import or execute candidates and does not repair the historical task-contract or timing defects.

- `20260520_202314`: 11/20 preserved candidates pass the current strict policy.
- `20260520_213128`: 6/8 preserved candidates pass the current strict policy.
- Free-function entry points: 28/28 preserved candidates.

Current-policy rejection reasons:

- `import_time_call:ConvTranspose3DWrapper.cuda.eval`: 1
- `obvious_torch_fallback:F.conv2d`: 1
- `obvious_torch_fallback:F.conv3d`: 1
- `obvious_torch_fallback:loss.mean`: 1
- `obvious_torch_fallback:out.mean`: 1
- `obvious_torch_fallback:out.sum`: 1
- `obvious_torch_fallback:tensor_method:A.view`: 1
- `obvious_torch_fallback:torch.log`: 1
- `obvious_torch_fallback:torch.nn.functional.conv2d`: 1
- `obvious_torch_fallback:torch.nn.functional.conv3d`: 1
- `obvious_torch_fallback:torch.nn.functional.conv_transpose2d`: 1

## Interpretation

The historical `policy pass` counts cannot be compared directly with current policy counts because the policy implementation changed. Even a source that passes the current AST guardrail is not thereby valid for an official KernelBench `Model` task without the `ModelNew` contract, and AST checks are not a security sandbox. The machine-readable rows are in `reports/tables/kernelbench_historical_policy_reaudit.csv`.
