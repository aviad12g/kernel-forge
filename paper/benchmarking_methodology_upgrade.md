# Benchmarking Methodology Upgrade Plan

This document defines the upgraded measurement path. It is implemented as an opt-in benchmark mode and is not yet the default for all legacy configs.

## Goals

The benchmarker should reduce false performance wins and make timing evidence strong enough for a KernelBench L1 repeatability study.

## Methodology

1. Use `torch.cuda.Event` timing for GPU kernels.
2. Run warmup iterations before measurement.
3. Use at least 100 samples for final reported candidates.
4. Report median and interquartile range.
5. Report confidence intervals or bootstrap confidence intervals where practical.
6. Add L2/cache flushing between samples using a large buffer write.
7. Separate compile time from runtime.
8. Separate candidate generation failures, compile failures, correctness failures, and performance failures.
9. Repeat top-k candidates across independent measurement sessions.
10. Mark unstable wins separately from repeat-stable wins.

The wall-clock timer remains available for CPU tests and development. Real CUDA/Triton performance reports should prefer CUDA-event timing because host-side wall-clock timing can mix kernel runtime with dispatch latency, Python scheduling, and synchronization behavior.

## Report Fields

Rigorous benchmark reports include:

- candidate median runtime
- IQR
- min/max
- standard deviation
- coefficient of variation
- number of samples
- number of warmup iterations
- independent session count
- repeat median
- repeat IQR
- coefficient of variation
- stable-win label
- single-run-only label
- torch.compile baseline runtime if available
- cache flush enabled/performed
- compile-time measurement or a warning when unavailable

## Methodology Check

The sanity-check command exercises the rigorous path on a tiny subset:

```bash
python -m openkernelforge.cli benchmark-methodology-check \
  --config configs/template_fused8_gpu_benchmark_rigorous.yaml \
  --max-tasks 2
```

On CPU-only machines this exits cleanly with a skipped report. On CUDA machines it confirms CUDA-event timing, optional cache flushing, sample summaries, and clean handling of `torch.compile` results or failures.

## Implementation Status

- CUDA-event timing: implemented.
- CPU fallback and CPU tests: implemented and passing.
- Optional CUDA cache flushing: implemented and reported when enabled.
- Independent sessions and richer sample summaries: implemented.
- CUDA methodology check: pending in this local checkout because the machine is CPU-only.
- Rigorous small fused8 validation: config exists at `configs/template_fused8_gpu_benchmark_rigorous_small.yaml`; CUDA RunPod execution is pending.
- Full fused8 rerun with rigorous timing: pending. Legacy fused8 tables should remain labeled as legacy timing until regenerated.

## Acceptance Criteria For A Claimed Win

A generated kernel should be called faster only if:

- it passes correctness
- it has runtime-only timing separated from compile time
- it beats the baseline by median runtime
- the win survives independent repeatability sessions
- it is not explained by a one-off outlier or unstable timing distribution

## Non-Goals For This Plan

- Do not add Nsight profiling in this sprint.
- Do not change task definitions while changing timing methodology.
- Do not run KernelBench before validating the upgraded benchmarker on internal fused8 tasks.
