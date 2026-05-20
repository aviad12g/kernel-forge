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

Current benchmark reports compare candidate runtime against PyTorch eager and optionally `torch.compile`. Reports distinguish:

- generation or extraction failures
- policy rejections
- Triton compile failures
- runtime failures
- correctness failures
- correct-but-slow candidates
- correct-and-fast candidates

The current benchmark implementation is sufficient for the internal fused8 study but should be upgraded before broader claims. See `benchmarking_methodology_upgrade.md`.

## Repeatability

Top candidates are rebenchmarked with repeated measurement sessions. Reports include median speedup, variability, coefficient of variation, and stability status. A candidate is not treated as a stable win unless repeatability confirms it.

## Dataset Curation

Dataset export separates rows by intended use:

- repeat-stable fast rows for reviewed SFT candidates
- single-run-only fast rows for analysis
- promising rows below eager for optimization/ranking research
- optimization pairs where one correct candidate beats another
- rejected or unstable rows for failure analysis

This separation prevents unstable single-run results from becoming direct training targets without review.
