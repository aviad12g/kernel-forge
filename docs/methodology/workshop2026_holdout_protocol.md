# Workshop 2026 Holdout-Confirmation Protocol

Status: completed. The executable protocol and task/candidate manifests were
prespecified and checksum-frozen before corrected candidate timing. This was an
internal prospective freeze, not a public preregistration.

## Research questions and evidence separation

1. **RQ1, repeatability:** under a fixed three-candidate budget, how many
   contract-valid KernelBench tasks, screening wins, and independently
   confirmed wins remain?
2. **RQ2, selection bias:** in a separate controlled fused8 study with 20
   frozen candidates per task and independent data for every candidate, how
   does selection optimism change with candidate budget?
3. **RQ3, evaluator validity:** in a separate disposable-process ablation, how
   much denominator inflation is caused by reconstructing and transferring the
   reference inside the measured call?

RQ2 is not inferred from the three-candidate RQ1 campaign. Its executable
specification is `docs/methodology/workshop2026_multiplicity_protocol.md`.

## Performance-blind task selection

The source pool is KernelBench L1 at commit
`423217d9fda91e0c2d67e4a43bf62f96f6d104f1`. The task freezer loads official
task source, materializes inputs for memory preflight, and computes the existing
verification-residency lower bound. It does not read candidates or timing data.
Tasks whose known lower bound exceeds 8 GiB are excluded explicitly. This is
not a peak-memory guarantee. The initial protocol requested 50 selected tasks.
Before manifest freeze and candidate generation, a performance-blind preflight
of the pinned 100-task pool found exactly 49 tasks at or below the 8 GiB cap;
the next memory tier began at 20 GiB. One feasible task is reserved by the same
deterministic order for the required excluded-task shakedown, leaving 48 primary
tasks. The cap, family order, and deterministic family round-robin selection
rule remain unchanged. Candidate correctness and performance fields were not read.
The JSON manifest, source hashes, and checksum are frozen before generation.

## Initialization and correctness

Official `Model` and candidate `ModelNew` implementations receive the same
frozen constructor arguments with the RNG restored to the same constructor
seed. This does not require cross-module state equality. Source and constructor
hashes establish provenance; before/after state snapshots establish that each
module state remains unchanged during verification and timing.

Correctness uses official `get_inputs()` with seeds `1103`, `2207`, `3301`,
`4409`, and `5519`. Each case checks:

- two executions on the same logical input for deterministic values and tree;
- exact nested output structure, shape, and dtype;
- output-to-input and output-to-output alias patterns;
- input-effect parity and no unexpected mutation;
- exact NaN, positive-infinity, and negative-infinity masks; and
- the official task `rtol` and `atol`.

The runtime audit rejects high-level ATen compute outside the allocation
allowlist and requires an observed Triton JIT launch. The campaign does not
claim generic nearby-shape, non-contiguous-layout, distribution-shift, or
high-precision accumulation oracles where the official task contract does not
define them.

## Screening and confirmation

Three candidates per task are generated under one frozen provider, model,
prompt, and decoding budget. Generation preserves the prompt, raw response,
extracted source, configured model string, provider-returned model field, and
checksums. It performs no performance evaluation.

Screening uses one process and 20 randomized paired blocks. The winner is the
valid candidate with the largest median block-level log speedup over eager;
candidate ID is the deterministic tie-breaker. The winner manifest is frozen
before confirmation. Tasks with no valid timed candidate remain `INVALID`.

Confirmation evaluates only the frozen winner in exactly seven new OS
processes, each with 20 paired blocks and fresh seeds. A missing process
invalidates the task and is not replaced. Processes `p00` through `p03` form
wave 1; `p04` through `p06` form wave 2. The runner requires separate
invocations, at least 30 minutes of separation, and the same GPU UUID and
software fingerprint. Wave-1 inspection is limited to integrity and
completeness. Analysis starts only after all seven artifacts exist.

For task `t`, candidate `c`, process `r`, and block `b`:

```text
z[t,c,r,b] = log(median_eager_ms / median_candidate_ms)
```

The task estimate is the median of process-level block medians. The process is
the resampling cluster, never an individual CUDA-event launch.

## Primary and secondary outcomes

The primary RQ1 report contains:

- selected task count, valid count, and invalid count;
- screening wins strictly above `1 + delta`;
- independently confirmed wins;
- false-promotion fraction among screening wins;
- median selection optimism in log space and ratio space;
- a two-sided 95% task-bootstrap percentile interval for median optimism; and
- p25, median, and p75 screen-to-confirm movement.

The practical margin is `delta = 0.02`. A secondary strict task-level analysis
forms a one-sided 95% percentile lower bound from exactly seven process
clusters, computes a centered one-sided bootstrap p-value under `log(1.02)`,
counts equality in the null tail, uses 20,000 draws with frozen seeds, and
applies Benjamini-Hochberg at `q = 0.05`. This secondary labeling does not
replace the aggregate outcomes.

## Timing and formal controls

Steady-state timing uses randomized paired CUDA-event blocks. Launch count is
adapted to at least 1.5 ms per interval. Cache state is perturbed before each
method using the same persistent read-write buffer, sized to twice reported L2
and clamped to 32--512 MiB. This is a practical cache-state perturbation, not a
guarantee that every cache level is completely flushed. A `torch.compile
max-autotune` baseline is materialized during screening and compile/first-call
cost is kept separate from runtime.

Seven fresh calibration processes run 20 blocks each. The null control compares
persistent eager with an identical persistent wrapper. The known-slowdown
control adds calibrated GPU work. The calibration passes only if:

- both controls have exactly seven process records;
- null median ratio is in `[0.995, 1.005]`;
- the null does not cross the 2% practical margin in either direction; and
- detected slowdown is in `[0.02, 0.08]`.

The historical reconstruct-per-call path is measured only in the separate
lifecycle control: the first selected task per represented family, capped at
eight tasks, with three disposable processes per task and ten blocks each,
synchronized host end-to-end timing, enclosing CUDA events, and a decomposition
of constructor, transfer, forward-enqueue, synchronization, and forward event
time where available. Enclosing events are explicitly not interpreted as pure
device compute when host lifecycle work occurs between them.

`scripts/check_campaign_validity.py` combines calibration and lifecycle
summaries. The screening runner requires a checksummed `PASS` gate for the same
protocol. Missing or failed controls prohibit candidate performance claims.

## Artifacts

Raw block times, process IDs, randomized method order, input hashes, candidate
hashes, correctness records, precision settings, GPU UUID, clocks, power,
temperature, driver, framework versions, and control summaries are preserved.
Historical KernelBench rows produced by the invalid free-function adapter are
audit artifacts only and are excluded from corrected performance analysis.
