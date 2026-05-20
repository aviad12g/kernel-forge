# Methodology

## Task Interface

Each task defines a PyTorch reference, deterministic input generation, benchmark shapes, dtype metadata, tolerances, and prompt hints. A candidate is valid only if it exposes:

```python
def forward(*args):
    ...
```

## Candidate Generation

OpenKernelForge supports model-generated and deterministic template-generated candidates:

- fake and dummy agents for offline tests
- OpenAI-compatible backends for local or remote model servers
- deterministic Triton template agents for baseline sweeps

Model prompts, raw responses, extracted candidates, and error logs are saved as run artifacts.

## Static Policy Check

Before verification, candidates are checked for basic policy compliance:

- `forward` must exist
- obvious PyTorch fallback is rejected when fallback mode is disabled
- reference imports and suspicious reference-call patterns are rejected
- uncertain issues are warnings rather than hard failures

Policy failures are not benchmarked.

## Correctness Verification

Candidates are executed against deterministic PyTorch references. The verifier records shape mismatches, dtype mismatches, numerical mismatches, NaN/Inf issues, exceptions, and tracebacks. A candidate must pass verification before benchmarking.

## Benchmarking

Benchmark reports compare candidate runtime against PyTorch eager and optionally `torch.compile`. The legacy path used process wall-clock timing and is retained for CPU development and older artifacts. The rigorous path is opt-in and uses CUDA events for GPU runtime timing so host scheduling overhead is not treated as kernel execution time.

The upgraded benchmark configuration records:

- timing mode: `cuda_event`, `wall_clock`, or `auto`
- warmup iterations before measurement
- measured sample count
- independent measurement sessions
- optional cache flushing between samples
- median, IQR, min/max, standard deviation, and coefficient of variation
- optional bootstrap confidence intervals
- runtime-only measurements separate from compile/setup timing where practical

Warmup matters because both CUDA kernels and PyTorch baselines can include lazy initialization. Compile time must be separated from runtime because generated kernels should be compared as kernels, not as compiler invocations. Cache flushing is optional because it changes the workload model; when enabled, reports must state that it was performed.

Reports distinguish:

- generation or extraction failures
- policy rejections
- Triton compile failures
- runtime failures
- correctness failures
- correct-but-slow candidates
- correct-and-fast candidates

Single-run timing is treated as a search signal, not a benchmark claim. The project should use CUDA-event timing plus repeatability sessions before making any broader performance claim. See `benchmarking_methodology_upgrade.md`.

## Repeatability

Top candidates are rebenchmarked with repeated measurement sessions. Reports include per-session median speedup, IQR, coefficient of variation, cache-flush status, and a conservative label: `REPEAT_STABLE_WIN`, `SINGLE_RUN_ONLY_WIN`, `UNSTABLE`, `BELOW_EAGER`, or `INSUFFICIENT_DATA`. A candidate is not treated as a stable win unless repeatability confirms it.

## Dataset Curation

Dataset export separates rows by intended use:

- repeat-stable fast rows for reviewed SFT candidates
- single-run-only fast rows for analysis
- promising rows below eager for optimization/ranking research
- optimization pairs where one correct candidate beats another
- rejected or unstable rows for failure analysis

This separation prevents unstable single-run results from becoming direct training targets without review.
