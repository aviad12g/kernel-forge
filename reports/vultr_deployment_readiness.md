# Vultr Deployment Readiness (Superseded)

Status: no Vultr deployment is required for the current workshop paper. The
corrected campaign completed on a Tesla T4, stayed within the five-hour budget,
and was imported under `artifacts/workshop2026/`. This document is retained as
the audit record for a deployment path that was prepared but never used.

Rechecked on 2026-08-21. The Vultr portal is authenticated but still displays
`Additional information required` and blocks deployment until account
verification is completed. The instance list is empty. No service was created
and no compute billing started.

## Approved deployment envelope

- Provider: Vultr
- Region: Singapore
- GPU plan: two full NVIDIA A16 GPUs, 16 GB each
- Image: Ubuntu 22.04 LTS GPU Enabled
- Automatic backups: disabled
- Last observed price: USD 0.942 per hour
- Hard spend cap: USD 70, approximately 74 hours at the observed rate
- Registered SSH key label: `openkernelforge-vultr`

The A16 plan is a fallback selected after full A40, L40S, A100, and single GH200
plans were unavailable in the portal. Availability and price must be rechecked
at deployment time.

## Prepared execution package

- Bundle: `artifacts/openkernelforge_workshop2026_gpu_bundle.tar.gz`
- Bundle size: approximately 342 KB
- Bundle SHA-256: `0fc11f1cad8918524d585db800d9d67e106e98d7bec08bb54e276fd2941e37d5`
- Bundle validation: secret scan passed; embedded `SHA256SUMS` verified after
  extraction
- Baseline compatibility entrypoint: `scripts/run_corrected_cuda_campaign.py`
- Workshop protocol entrypoints: `scripts/freeze_kernelbench_task_selection.py`,
  `scripts/run_evaluator_controls.py`, and
  `scripts/run_holdout_confirmation_campaign.py`
- Official KernelBench commit:
  `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`
- Default wall-time limit: five hours
- Stages: corrected five-task baseline gate, then corrected 20-task safe baseline
- Candidate generation: disabled

## Remaining external gate

The account owner must complete Vultr identity/account verification. Do not store
government ID, phone, payment details, or verification responses in this
repository. After the portal removes the block, recheck plan availability and
price before creating the instance. A successful order receipt and a running
instance must both be observed before reporting deployment as complete.

The corrected baseline campaign validates adapter and eager/compiler timing only.
The workshop holdout protocol additionally requires a performance-blind frozen
48-task manifest and three new corrected `ModelNew` candidates per selected
task. Historical free-function candidates cannot restore a KernelBench claim.
The Vultr identity gate was never crossed and no Vultr compute was created or
billed. The previously missing corrected candidate manifest now exists from the
completed T4 campaign at `artifacts/workshop2026/candidate_manifest.json`.
