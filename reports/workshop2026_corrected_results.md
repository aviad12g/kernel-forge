# Workshop 2026 Corrected Results

Status: completed, checksummed, and included in the formal workshop paper.

## Primary holdout result

The official KernelBench L1 pool was pinned to commit
`423217d9fda91e0c2d67e4a43bf62f96f6d104f1`. Performance-blind preflight
selected 48 tasks under the fixed 8 GiB known-residency cap, using deterministic
family round-robin ordering. Three Gemini candidates were frozen for each task
before any timing.

- Selected tasks: 48.
- Generated candidates: 144.
- Candidate records reaching evaluation: 141. One task's compiler baseline
  exhausted memory before its three candidates could be evaluated.
- Evaluated-candidate breakdown: 77 static-policy failures, 0 contract-only
  failures, 9 numerical or repeat-correctness failures, 28 candidate
  compile/runtime failures, and 27 full-gate passes.
- Candidates passing static policy, five-seed contract/correctness checks,
  runtime auditing, and paired timing: 27, covering 10 tasks.
- Frozen valid task winners: 10, all matrix-multiplication tasks.
- Compiler baselines available: 47/48 selected tasks and 10/10 frozen valid-task
  winners during screening. One winner cleared the 2% compiler margin
  (1.826x) while remaining below eager (0.937x). Confirmation did not rerun the
  compiler baseline.
- Screening wins above the prespecified 2% eager margin: 0.
- Independently confirmed wins above the margin: 0.
- Median screening-to-confirmation ratio: 1.013656.
- Task-bootstrap 95% interval: [0.915082, 1.170396].
- False-promotion fraction: undefined because no screening winner crossed the
  promotion margin.

The corrected external campaign therefore supports a validity and
re-evaluation result, not a generated-kernel speedup claim.

## Controlled multiplicity result

The separate fused8 study froze 20 deterministic candidates for each of four
tasks and confirmed every valid candidate in seven fresh processes. At every
candidate budget in `{1, 2, 3, 5, 10, 20}`, both apparent and independently
confirmed win rates were 1.0. Median log selection optimism ranged from
-0.000842 to 0.000004. The derived paper figure reports post-hoc task-bootstrap
intervals from the preserved all-candidate timing blocks; the frozen campaign
CSV and checksum ledger remain unchanged.

This is a boundary condition: the deterministic grid did not exhibit
multiplicity-driven false promotion. It does not establish that candidate
search is harmless in other generators or task families.

A same-GPU replication reran the identical frozen grid on the RTX A4500 used by
the accepted near-threshold campaign. All 32 worker records completed and the
checksum ledger verifies. Apparent and confirmed win rates remained 1.0 at
every budget through `K=20`; at `K=20`, median log optimism was `-0.000112`
with interval `[-0.001038, 0.001724]`. This removes the easy-grid hardware
confound without broadening the four-task scope.

## Calibrated near-threshold multiplicity stress test

A separate RTX A4500 campaign froze 20 delayed deterministic variants for each
of the same four fused8 tasks. Three disjoint calibration processes selected
eight candidates per task inside the prespecified `[0.98, 1.04]` window. The
calibration data were excluded from primary screening and confirmation. All 32
selected candidates then completed one 20-block screening process and seven
fresh 20-block confirmation processes after a 30-minute separation.

- At `K=1`, apparent and confirmed win rates were 0.1536 and 0.1243.
- At `K=8`, apparent and confirmed win rates were 0.7500 and 0.5000.
- The full-budget screening winners were apparent wins for three of four tasks;
  two remained above the prespecified 2% margin in confirmation.
- `bias_gelu` was the observed screen-only promotion: 1.0271x in screening and
  1.0001x in independent confirmation.
- Median log optimism at `K=8` was 0.007248 with four-task bootstrap interval
  `[-0.012395, 0.026614]`.

The stress test therefore demonstrates the false-promotion mechanism in one
calibrated regime. The interval contains zero and only four tasks were studied,
so it does not estimate a population effect or the natural prevalence of
near-threshold candidates. Two earlier calibration grids were retained as
design provenance and did not advance to primary screening.

## Evaluator controls

- Excluded-task shakedown: PASS, including static policy, five-seed
  correctness, runtime auditing, observed Triton launch, and paired timing.
- Calibration: PASS in seven processes. The null wrapper ratio was 0.999998;
  the known-slowdown ratio was 0.946687, corresponding to a detected slowdown
  fraction of 0.056316.
- Lifecycle ablation: PASS for 24/24 process rows. Reconstructing and
  transferring the reference inside the measured call inflated synchronized
  host latency by a median factor of 1.053, while the median enclosing
  CUDA-event ratio was 1.000.

## Artifact integrity

- Holdout checksum ledger: `artifacts/workshop2026/holdout_campaign/SHA256SUMS`
  (249 entries).
- Multiplicity checksum ledger:
  `artifacts/workshop2026/multiplicity/campaign/SHA256SUMS` (67 entries).
- Near-threshold checksum ledger:
  `artifacts/workshop2026/near_threshold_multiplicity_v3/campaign/SHA256SUMS`
  (94 entries).
- Complete imported checkpoint:
  `artifacts/colab_checkpoints/okf_checkpoint_gpu_complete_v1_10.tar.gz`.
- Checkpoint SHA-256:
  `9174cd67ebe2b0bde4a59b1952383c6d30dd7c6b7b2702a219df05038846a02f`.
- Recorded holdout plus multiplicity worker time: 2.091 GPU-hours on a Tesla T4.
- Recorded accepted near-threshold v3 worker time: 0.368 GPU-hours on an RTX
  A4500. Calibration-only v1 and v2 design pilots used about 0.255 additional
  recorded worker-hours and produced no primary estimates.

The historical 20-task adapter pilot remains an evaluator-audit artifact. Its
old correctness, speed, profiler, and clock rows are not merged with the
corrected campaign.

## Compiler-rung confirmation

The one frozen task winner that beat the compiler at screening was rechecked
against `torch.compile max-autotune` in seven fresh RTX A4500 processes. The
median candidate-versus-compile ratio was `2.001165x`; process medians ranged
from `1.993307x` to `2.003077x`. Compile-and-first-call latency was recorded
separately. This control confirms the compiler rung for one candidate and does
not alter its primary `0.937x` below-eager screening result.
