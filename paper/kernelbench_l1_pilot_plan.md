# KernelBench L1 Repeatability Pilot Plan (Historical)

This file records the original KernelBench L1 pilot plan and is retained for
provenance. A post-hoc audit found that the run used a free-function candidate
contract instead of official `ModelNew` instances and rebuilt reference modules
inside timed calls. Consequently, all KernelBench correctness, repair, timing,
profiler, and clock-validation rows below are affected-evaluator output only.
They are not supported external results. The corrected adapter is implemented
and CPU-tested. This historical plan is superseded by the completed corrected
workshop campaign documented in `reports/workshop2026_corrected_results.md`.

## Objective

Measure how often apparent single-run speedups from generated kernels remain valid under repeatability-aware benchmarking.

Historical run hypothesis:

> Test whether a capped external-task pilot can exercise generation,
> verification, timing, failure analysis, and repair.

## Task Set

Start with 20 KernelBench L1 tasks. Select tasks that:

- can be represented through the OpenKernelForge task interface
- have deterministic input generation
- have clear PyTorch reference implementations
- are small enough for repeated local GPU benchmarking
- cover a mix of elementwise fusion, reductions, normalization, and simple matmul-style patterns

## Protocol

1. Import or adapt 20 KernelBench L1 tasks into OpenKernelForge.
2. Run PyTorch eager and optional `torch.compile` baselines with the rigorous CUDA-event path.
3. Validate task loading, input generation, tolerances, and baseline timing before candidate generation.
4. Generate candidates with a capped existing OpenKernelForge provider path.
4. Apply static policy checks.
5. Verify correctness against PyTorch references.
6. Benchmark correct candidates with the upgraded CUDA-event methodology.
7. Identify single-run winners.
8. Repeat top-k candidates across independent measurement sessions.
9. Report stable winners separately from unstable single-run winners.

## Current Implementation Status

The internal fused8 benchmark now has a rigorous CUDA-event deterministic-template run:

- Run: `runs/20260520_155839`
- Candidates: 160
- Verified: 160/160
- Repeat-stable wins: `residual_add_relu`, `bias_gelu`, `rmsnorm_small`

The first KernelBench L1 sprint added a local adapter and baseline-validation
command. The affected evaluator then ran a capped Gemini candidate pilot and
one repair iteration on a feasible 20-task subset:

- Official KernelBench checkout: `/workspace/KernelBench`
- Commit: `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`
- Safe 20-task baseline validation: `/workspace/openkernelforge/runs/20260520_181052`
- Capped Gemini candidate pilot: `/workspace/openkernelforge/runs/20260520_202314`
- Capped Gemini repair pass: `/workspace/openkernelforge/runs/20260520_213128`
- Historical evaluator verification fields: 3/20 one-shot and 1/8 repair
- Historical evaluator labels: `CrossEntropyLoss`, `TripletMarginLoss`, and `KLDivLoss`
- Evidence status: invalid for model accuracy, repair effectiveness, or speed

```bash
python -m openkernelforge.cli kernelbench-l1-check \
  --config configs/kernelbench_l1_5task_rigorous.yaml \
  --kernelbench-dir <path-to-kernelbench>
```

The current command validates local KernelBench task loading, persistent PyTorch
references, official `ModelNew` candidates, optional `torch.compile`, and timing
metadata. The Gemini and repair configs add capped candidate generation. No
KernelBench correctness or speed result is supported until corrected CUDA
revalidation completed in the separate workshop campaign.

## Primary Metric

Repeat-stable speedup rate:

```text
number of tasks with at least one repeat-stable faster candidate
/
number of evaluated tasks
```

## Secondary Metrics

- correctness rate
- compile failure rate
- runtime failure rate
- single-run win rate
- single-run win decay rate
- median speedup among verified candidates
- unstable-win fraction
- policy rejection rate
- repeatability coefficient of variation

## Required Outputs

- run artifacts with prompts, responses, candidates, logs, results, and environment probe
- per-task benchmark summary
- repeatability report
- failure taxonomy table
- single-run-vs-repeat-stable comparison table
- curated dataset export with stable and unstable rows separated

## Stop Conditions

Stop the pilot and fix methodology if:

- CUDA-event timing disagrees strongly with existing benchmarker on internal fused8 sanity checks
- repeatability reports cannot be reproduced across sessions
- task imports silently alter semantics
- any API key or secret appears in artifacts

## Current Claim Boundary

Allowed now:

- "The preserved historical run exercised the artifact and failure-analysis paths."
- "A post-hoc audit found candidate-contract and reference-lifecycle defects."
- "The corrected adapter requires official `ModelNew` instances and persistent seeded references."
- "Corrected CUDA revalidation is reported separately; historical rows remain excluded."

Not allowed:

- SOTA claims
- any KernelBench correctness, repair-effectiveness, or speedup claim from the affected runs
- broad or full KernelBench claims
- training claims
- claims based only on single-run speedups
