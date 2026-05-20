# KernelBench L1 Repeatability Pilot Plan

This is a plan, not an executed KernelBench result.

## Objective

Measure how often apparent single-run speedups from generated kernels remain valid under repeatability-aware benchmarking.

Proposed headline:

> X% of single-run wins fail repeat verification.

`X` is intentionally blank until the pilot is run.

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
4. Generate candidates with the existing OpenKernelForge model/template path in a later sprint.
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

The first KernelBench L1 sprint adds a local adapter and baseline-validation command only:

```bash
python -m openkernelforge.cli kernelbench-l1-check \
  --config configs/kernelbench_l1_5task_rigorous.yaml \
  --kernelbench-dir <path-to-kernelbench>
```

This command validates local KernelBench task loading, PyTorch eager references, optional `torch.compile`, and timing metadata. It does not run LLM generation. No KernelBench performance result should be claimed until real KernelBench tasks and candidates are evaluated.

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

## Claims Allowed After Pilot

Allowed:

- "On a 20-task KernelBench L1 pilot, X% of single-run wins failed repeat verification."
- "Repeatability-aware evaluation changed model/template ranking on these tasks."

Not allowed:

- SOTA claims
- broad KernelBench claims beyond the 20-task pilot
- training claims
- claims based only on single-run speedups
