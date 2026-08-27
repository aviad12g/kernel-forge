# Workshop 2026 Controlled Multiplicity Protocol

Status: completed. The candidate population and analysis protocol were
prespecified and checksum-frozen before timing; they were not publicly
preregistered.

## Purpose

This study isolates candidate-budget selection optimism. It is separate from
the three-candidate KernelBench holdout study because a budget curve is not
identifiable when only the selected candidate receives independent timing.

## Frozen candidate population

The study uses four controlled fused8 tasks at the existing `[4096, 1024]`
float32 shape regime:

- `bias_relu`
- `residual_add_relu`
- `bias_gelu`
- `rmsnorm_small`

For each task, `TemplateAgent` emits exactly 20 deterministic variants from the
frozen search grid in `configs/workshop2026_multiplicity_protocol.yaml`. Source,
metadata, protocol hash, and candidate-manifest hash are written before any
timing. The freezer refuses overwrite.

## Correctness and timing

Every candidate receives the same five-seed correctness matrix, same-input
determinism check, exact output tree, alias contract, static policy, and runtime
Triton/ATen audit as the corrected campaign. Each valid candidate receives:

- 20 randomized paired screening blocks in one process; and
- 20 randomized paired blocks in each of seven fresh confirmation processes.

Thus confirmation availability cannot depend on which candidate wins a
resample. A task is excluded at budget `K` if it has fewer than `K` valid
candidates with all seven process records. No process replacement is allowed.

## Analysis

For each `K` in `{1, 2, 3, 5, 10, 20}`, 1,000 deterministic task-level
resamples draw `K` candidates without replacement, choose the largest screening
median log speedup, and evaluate that same candidate on its already collected
confirmation data. The report contains:

- eligible task and candidate counts;
- apparent win rate above the fixed 2% margin;
- confirmed point-estimate win rate above the same margin; and
- median screening-minus-confirmation log speedup.

The RQ2 curve is descriptive. It does not repeat thousands of per-resample BH
tests; the strict process-cluster/BH promotion analysis belongs to RQ1. The
paper must present valid candidate counts and all-candidate confirmation
completeness alongside the curve.

## Boundary

Deterministic templates make the candidate population reproducible and avoid
API cost, but they do not estimate LLM sampling behavior. The KernelBench study
measures corrected external-task promotion under a fixed three-candidate LLM
budget; this study measures budget-induced selection optimism in a controlled
template population. Neither result is a benchmark-wide performance claim.
