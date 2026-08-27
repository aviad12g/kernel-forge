# Workshop 2026 GPU Handoff

Status: completed on a Tesla T4. Corrected task selection, candidate freezing,
controls, screening, two confirmation waves, controlled multiplicity, analysis,
and checksum ledgers are present under `artifacts/workshop2026/`. Historical
KernelBench rows remain audit-only.

Recorded holdout plus multiplicity worker time was 2.091 GPU-hours. The final
imported checkpoint is
`artifacts/colab_checkpoints/okf_checkpoint_gpu_complete_v1_10.tar.gz` with
SHA-256
`9174cd67ebe2b0bde4a59b1952383c6d30dd7c6b7b2702a219df05038846a02f`.

Protocol amendment (2026-08-27, before manifest freeze and candidate
generation): the primary selected-task target is 48 rather than 50.
Performance-blind preflight of the pinned 100-task pool found exactly 49 tasks
at or below the unchanged 8 GiB known-peak cap; the next tier began at 20 GiB.
One feasible task is reserved by the same deterministic order for the required
excluded-task shakedown. Candidate and performance fields were not read.

## Hard gates

1. Linux CUDA/Triton reports `TRITON_EXECUTION_OK`.
2. KernelBench is at commit
   `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`.
3. The task manifest is frozen before candidate generation.
4. Candidate generation preserves prompts, responses, sources, configured and
   provider-returned model metadata, and checksums.
5. Calibration and isolated lifecycle controls pass before screening.
6. Confirmation runs as two invocations separated by at least 30 minutes.
7. No outcome analysis occurs after wave 1.

## Execution order

```bash
cd /workspace/openkernelforge
python -m pip install -e '.[dev,paper,triton]'
python -m openkernelforge.cli env-check

git clone https://github.com/ScalingIntelligence/KernelBench.git /workspace/KernelBench
git -C /workspace/KernelBench checkout 423217d9fda91e0c2d67e4a43bf62f96f6d104f1

python scripts/freeze_kernelbench_task_selection.py \
  --protocol configs/workshop2026_holdout_protocol.yaml \
  --kernelbench-dir /workspace/KernelBench

python scripts/freeze_multiplicity_candidates.py \
  --protocol configs/workshop2026_multiplicity_protocol.yaml

python scripts/run_workshop2026_shakedown.py \
  --protocol configs/workshop2026_holdout_protocol.yaml \
  --task-manifest artifacts/workshop2026/task_selection_manifest.json \
  --kernelbench-dir /workspace/KernelBench

export GEMINI_API_KEY=<key>
python scripts/generate_workshop2026_candidates.py \
  --protocol configs/workshop2026_holdout_protocol.yaml \
  --task-manifest artifacts/workshop2026/task_selection_manifest.json \
  --kernelbench-dir /workspace/KernelBench \
  --allow-api-generation
unset GEMINI_API_KEY

python scripts/run_evaluator_controls.py \
  --protocol configs/workshop2026_holdout_protocol.yaml \
  --output-dir artifacts/workshop2026/evaluator_controls

python scripts/run_lifecycle_ablation.py \
  --protocol configs/workshop2026_holdout_protocol.yaml \
  --task-manifest artifacts/workshop2026/task_selection_manifest.json \
  --kernelbench-dir /workspace/KernelBench \
  --output-dir artifacts/workshop2026/lifecycle_ablation

python scripts/check_campaign_validity.py \
  --protocol configs/workshop2026_holdout_protocol.yaml \
  --calibration-validity artifacts/workshop2026/evaluator_controls/calibration_validity.json \
  --lifecycle-summary artifacts/workshop2026/lifecycle_ablation/lifecycle_ablation_summary.json \
  --output artifacts/workshop2026/campaign_validity.json

python scripts/run_holdout_confirmation_campaign.py \
  --protocol configs/workshop2026_holdout_protocol.yaml \
  --task-manifest artifacts/workshop2026/task_selection_manifest.json \
  --candidate-manifest artifacts/workshop2026/candidate_manifest.json \
  --kernelbench-dir /workspace/KernelBench \
  --campaign-validity artifacts/workshop2026/campaign_validity.json \
  --output-dir artifacts/workshop2026/holdout_campaign \
  --max-gpu-hours 5 \
  --screen-only

python scripts/run_holdout_confirmation_campaign.py \
  --task-manifest artifacts/workshop2026/task_selection_manifest.json \
  --candidate-manifest artifacts/workshop2026/candidate_manifest.json \
  --kernelbench-dir /workspace/KernelBench \
  --campaign-validity artifacts/workshop2026/campaign_validity.json \
  --confirmation-wave wave1 \
  --max-gpu-hours 5

# Integrity/completeness review only. Wait until wave1_integrity_lock.json permits wave 2.
python scripts/run_holdout_confirmation_campaign.py \
  --task-manifest artifacts/workshop2026/task_selection_manifest.json \
  --candidate-manifest artifacts/workshop2026/candidate_manifest.json \
  --kernelbench-dir /workspace/KernelBench \
  --campaign-validity artifacts/workshop2026/campaign_validity.json \
  --confirmation-wave wave2 \
  --max-gpu-hours 5

python scripts/run_multiplicity_campaign.py \
  --protocol configs/workshop2026_multiplicity_protocol.yaml \
  --candidate-manifest artifacts/workshop2026/multiplicity/candidate_manifest.json \
  --output-dir artifacts/workshop2026/multiplicity/campaign \
  --max-gpu-hours 5
```

Before candidate generation, inspect
`artifacts/workshop2026/shakedown_excluded_task/shakedown_summary.json`.
The shakedown must preserve paired CUDA-event blocks and telemetry on one
feasible task outside the frozen selection. Its separate deterministic fused8
candidate gate must also pass static policy, five-seed correctness, runtime
ATen auditing, and observed Triton launch checks. Every shakedown artifact is
marked ineligible for paper results.

The holdout runner never calls an API. It refuses a missing or failed campaign
gate, a changed winner manifest, a collapsed temporal wave, a stale protocol
hash, or an incomplete seven-process task. The multiplicity runner separately
times and confirms all 20 deterministic candidates per controlled fused8 task.

## Completed evidence for paper promotion

- formal calibration and lifecycle `PASS` records;
- valid, invalid, screening-win, and confirmed-win counts;
- false-promotion fraction;
- selection-optimism median with task-bootstrap interval;
- screen-to-confirm movement distribution;
- strict per-task cluster-bootstrap/BH labels as secondary analysis;
- all-candidate multiplicity curve with validity/completeness counts;
- environment, clock, power, input, source, and manifest hashes; and
- a submission build with no pending-evidence marker.

All required control gates passed before screening. The holdout ledger contains
249 entries and the multiplicity ledger contains 67 entries. The traced outcome
summary is `reports/workshop2026_corrected_results.md`; the formal strict build
is `paper/workshop2026/openkernelforge_workshop2026.pdf`.
