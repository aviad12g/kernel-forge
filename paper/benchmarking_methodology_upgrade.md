# Benchmarking Methodology Upgrade Plan

This is a plan for the next measurement sprint. It is not yet implemented as the default benchmark path.

## Goals

The benchmarker should reduce false performance wins and make timing evidence strong enough for a KernelBench L1 repeatability study.

## Planned Methodology

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

## Report Fields

Future reports should include:

- candidate median runtime
- IQR
- min/max
- number of samples
- number of warmup iterations
- independent session count
- repeat median
- repeat IQR
- coefficient of variation
- stable-win label
- single-run-only label
- torch.compile baseline runtime if available

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
