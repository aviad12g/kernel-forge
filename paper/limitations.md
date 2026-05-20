# Limitations

- The current results are from an internal fused8 benchmark, not KernelBench.
- The task set is small and controlled.
- The reported GPU campaign used limited hardware coverage.
- Timing uses the current harness benchmarker; the next methodology sprint should upgrade measurement with CUDA events, more samples, IQR, and confidence intervals.
- No Nsight or hardware-counter profiling is included yet.
- The project has not trained or fine-tuned a model.
- The project has not added RL or execution-feedback optimization.
- API-model results were cheap-budget baselines, not exhaustive searches.
- Qwen 14B has no result because serving failed due disk/cache capacity.
- Full RunPod artifacts must be preserved and validated locally before public artifact claims.
- No SOTA claim is made.

## Risk Of Misinterpretation

The main risk is over-reading single-run benchmark wins. The project explicitly separates:

- single-run fast candidates
- repeat-stable fast candidates
- promising but slow candidates
- failed or unstable candidates

Only repeat-stable candidates should be discussed as benchmark wins.
