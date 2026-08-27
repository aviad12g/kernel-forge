# Corrected CUDA Campaign

## Purpose

The corrected campaign validates the official KernelBench L1 adapter and the
eager/`torch.compile` timing baselines after the `ModelNew` lifecycle fix. It is
baseline-only. It does not generate candidates and cannot support a generated
kernel performance claim.

## Fixed provenance

- KernelBench repository: `https://github.com/ScalingIntelligence/KernelBench`
- Required commit: `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`
- Smoke config: `configs/kernelbench_l1_5task_corrected_rigorous.yaml`
- Pilot config: `configs/kernelbench_l1_20task_corrected_rigorous_safe.yaml`
- Candidate provider: `none`
- Timing: CUDA events, 30 warmups, 120 measured samples per session
- Sessions: three same-process loops with rotating eager/reference-self/compile order
- Cache-state perturbation: 128 MB CUDA writes before measured samples
- Compiler baseline: `torch.compile` with `max-autotune`; wrapper and first-call
  compile cost recorded separately from runtime samples
- Memory preflight: metadata/meta-tensor materialization with a 3072 MB conservative
  known-memory cap; CPU materialization fallback is disabled

The memory estimate covers materialized inputs, reference state, verifier copies,
and known cache overhead. It remains a lower bound because temporary outputs,
compiler workspaces, allocator fragmentation, and backend-specific buffers are not
fully known before execution.

## Fail-closed sequence

`scripts/run_corrected_cuda_campaign.py` performs these gates in order:

1. Refuse Darwin, missing `nvidia-smi`, unavailable CUDA/Triton, or a failed tiny
   Triton kernel before writing campaign artifacts or cloning KernelBench.
2. Verify both configs are baseline-only and require CUDA-event timing, at least
   100 samples, three sessions, cache perturbation, and the compiler baseline.
3. Require the exact official KernelBench commit.
4. Run the five-task smoke stage.
5. Require five persistent `ModelNew` contracts, five eager timings, five compiler
   timings, cache perturbation for every task, and no failures.
6. Run the 20-task safe pilot only after the smoke gate passes.
7. Apply the same validation to all 20 selected tasks.
8. Write a source fingerprint, config hashes, command output, environment probe,
   stage validation, and `SHA256SUMS` under
   `artifacts/corrected_cuda_campaign/<timestamp>/`.

The campaign has a five-hour wall-time limit by default. A command that exceeds
the remaining budget is terminated by the subprocess timeout and the manifest is
marked failed.

## Remote commands

Build a secret-scanned source bundle locally:

```bash
python scripts/package_corrected_cuda_bundle.py \
  --out artifacts/openkernelforge_corrected_cuda_bundle.tar.gz
```

After transferring and extracting it on a disposable CUDA worker:

```bash
cd /workspace/openkernelforge
python scripts/run_corrected_cuda_campaign.py \
  --kernelbench-dir /workspace/KernelBench \
  --clone-kernelbench-if-missing \
  --max-wall-hours 5
```

No API key is required or permitted by these configs. A later candidate campaign
must use a new reviewed config and must not treat historical free-function
candidates as corrected `ModelNew` candidates.
