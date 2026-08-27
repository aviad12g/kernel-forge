# OpenKernelForge Reproducibility Guide

This guide reproduces the harness and internal fused8 workflow. Real Triton performance results require a CUDA GPU with Triton installed.

## 1. Install

```bash
python -m pip install -e .
pytest -q
```

## 2. Environment Check

```bash
python -m openkernelforge.cli env-check
```

For true Triton benchmark results, the viability should be `TRITON_EXECUTION_OK`.

## 3. Fake Smoke Run

```bash
python -m openkernelforge.cli smoke
```

Fake and dummy runs are harness checks only. They are not model benchmarks.

## 4. Fused8 Template Quick

```bash
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/template_fused8_gpu_autotune_quick.yaml \
  --out-name template_fused8_gpu_quick
```

## 5. Fused8 Template Wide

```bash
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/template_fused8_gpu_autotune_wide.yaml \
  --out-name template_fused8_gpu_wide
```

## 6. Repeatability

```bash
python -m openkernelforge.cli repeatability-report \
  --run-dir runs/<run> \
  --top-k 3 \
  --repeats 5
```

## 7. Optional Model Runs

Gemini/OpenAI runs require API keys in environment variables only. Do not commit keys.

```bash
export GEMINI_API_KEY=<your-key>
python scripts/run_gpu_baseline_3tasks.py --config configs/gemini_fused8_gpu_baseline.yaml --out-name gemini_fused8_gpu_baseline
unset GEMINI_API_KEY
```

OpenAI cheap runs use `OPENAI_API_KEY` and should be kept small unless early results justify more spend.

Local vLLM runs use the OpenAI-compatible local server path:

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --host 0.0.0.0 --port 8000
python scripts/run_local_model_fused8.py --config configs/qwen_fused8_gpu_baseline_cheap.yaml --out-name qwen_fused8_cheap
```

## 8. Curate Dataset

```bash
python -m openkernelforge.cli curate-fused8-dataset \
  --template-run runs/<template_run> \
  --gemini-run runs/<gemini_run> \
  --template-guided-run runs/<guided_run> \
  --out-dir datasets/fused8_curated_v1
```

## 9. Validate Curated Dataset

```bash
python -m openkernelforge.cli validate-curated-fused8 --dataset-dir datasets/fused8_curated_v1
```

## 10. Build Research Report

```bash
python scripts/build_paper_assets.py
python scripts/build_paper_pdf.py
python scripts/check_research_artifacts.py
```

## 10a. Methodology Definitions

The paper-facing methodology terms are specified in `docs/methodology/`.
These files are intended to make the evaluation reproducible without relying on
informal prose in the paper.

- `docs/methodology/repeatability_label_spec.md` defines the implemented
  repeatability labels and the difference between legacy repeatability-report
  labels and rigorous per-session labels.
- `docs/methodology/static_policy_checks.md` enumerates implemented AST policy
  checks before correctness verification.
- `docs/methodology/timing_protocol.md` specifies CUDA-event timing, warmup,
  samples, independent sessions, cache-state perturbation, and summary
  statistics.
- `docs/methodology/environment.md` records the known RunPod hardware/software
  environment and explicitly lists unrecorded clock, power, and thermal fields.
- `docs/methodology/prompt_templates.md` records representative fused8,
  KernelBench one-shot, and repair prompts plus decoding settings.
- `docs/methodology/fused8_tasks.md` lists fused8 shapes, dtypes, and
  correctness tolerances.
- `docs/methodology/kernelbench_repairability.md` defines the one-pass repair
  selection heuristic.
- `docs/methodology/kernelbench_adapter_contract.md` defines the corrected
  official `ModelNew` state and persistent-reference lifecycle.

Consistency checks:

```bash
python scripts/check_methodology_docs.py
python scripts/check_paper_text_clean.py
python scripts/check_pdf_text_clean.py
```

Existing-result statistical analysis:

```bash
python scripts/import_runpod_artifacts.py \
  --source-root /workspace/openkernelforge \
  --out-dir artifacts/runpod_imports
python scripts/analyze_existing_results_statistics.py
python scripts/analyze_kernelbench_interpretation.py
python scripts/analyze_label_threshold_sensitivity.py
python scripts/audit_historical_kernelbench_candidates.py
```

This imports any already-existing target RunPod artifacts, writes
`artifacts/runpod_imports/artifact_manifest.json` and `SHA256SUMS`, then writes
descriptive Wilson verification-rate intervals, KernelBench family-level
summaries, memory-filter summaries,
and single-run/repeat flip-frequency availability notes under `reports/tables/`,
`reports/statistical_notes.md`, `reports/fused8_artifact_recovery_notes.md`,
and `reports/kernelbench_interpretation_notes.md`. The KernelBench
interpretation script also reproduces historical repairability and family
summaries. Those KernelBench rows are now audit metadata because the source
run used an invalid state contract and reference lifecycle. The policy-audit
script parses preserved sources under the current `ast-v5` policy without
importing or executing them. It writes `reports/kernelbench_adapter_audit.md`
and `reports/tables/kernelbench_historical_policy_reaudit.csv`.

The candidate-level Gemini/OpenAI Fisher comparison was removed: three
candidates share each of eight fused8 tasks, so treating all outputs as
independent Bernoulli trials is not justified.

Headline clock validation uses the same preserved KernelBench loss candidates
and records clock, power, and temperature state before and after each small
validation block:

```bash
python scripts/validate_headline_clock.py \
  --kernelbench-dir /workspace/KernelBench \
  --try-lock-clocks
```

The preserved validation was run on a RunPod Flash RTX 4090 worker. It inherits
the historical adapter's invalid reference lifecycle, so it is a debugging
artifact rather than label-persistence evidence. Exact fused8 headline
candidate artifacts are not preserved locally. Outputs are
`reports/headline_clock_validation.md`,
`reports/tables/headline_clock_validation.csv`, and
`artifacts/headline_clock_validation/`.

The threshold-sensitivity script records whether preserved artifacts contain
the per-session vectors needed to recompute headline labels under alternate
`tau` values. In the current package, those vectors are not preserved, so no
threshold-robustness claim is made. Model identifiers are audited in
`reports/model_identifier_audit.md` and reported as configured API strings;
provider-returned model-version fields are not preserved for every run.

## 11. Corrected KernelBench Validation Workflow

The official KernelBench repository is pinned at commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`. Current code materializes one `get_init_inputs()` snapshot under seed 0 and constructs persistent reference `Model` and candidate `ModelNew` modules from it outside timed regions. Every official task that defines `Model` requires `ModelNew`; free functions are supported only by the local synthetic `reference_fn` interface. The memory estimate is a conservative lower bound over inputs, model state, and known harness overhead; it is not a peak-memory guarantee.

Clone KernelBench:

```bash
git clone https://github.com/ScalingIntelligence/KernelBench.git /workspace/KernelBench
cd /workspace/KernelBench
git checkout 423217d9fda91e0c2d67e4a43bf62f96f6d104f1
```

Run the corrected baseline-only campaign on CUDA. The wrapper refuses CPU/macOS
hosts, pins the official commit, runs a five-task gate before the 20-task pilot,
enforces a five-hour wall-time cap, and writes hashes plus a machine-readable
campaign manifest:

```bash
python scripts/run_corrected_cuda_campaign.py \
  --kernelbench-dir /workspace/KernelBench \
  --clone-kernelbench-if-missing \
  --max-wall-hours 5
```

The two provenance-specific configs are
`configs/kernelbench_l1_5task_corrected_rigorous.yaml` and
`configs/kernelbench_l1_20task_corrected_rigorous_safe.yaml`. Both use 120
samples per session, three sessions, CUDA events, 128 MB cache-state
perturbation, and `torch.compile` `max-autotune`. Build a secret-scanned source
bundle for a disposable worker with:

```bash
python scripts/package_corrected_cuda_bundle.py \
  --out artifacts/openkernelforge_workshop2026_gpu_bundle.tar.gz
```

Full gate semantics are specified in
`docs/methodology/corrected_cuda_campaign.md`. Passing this campaign validates
the corrected adapter and eager/compiler baselines only. It does not create or
validate generated candidates.

### Workshop holdout, controls, and multiplicity study

The submission study is separately frozen in
`configs/workshop2026_holdout_protocol.yaml` and
`configs/workshop2026_multiplicity_protocol.yaml`. Its exact ordered command
sequence is maintained in `reports/workshop2026_gpu_handoff.md`. The sequence
includes an excluded-task shakedown, task and candidate freezing, null and
known-slowdown calibration, an isolated reconstruct-per-call lifecycle control,
a formal `campaign_validity.json` gate, one screening invocation, two
confirmation waves separated by at least 30 minutes, and a separate
all-candidate fused8 multiplicity run.

The holdout runner requires:

```bash
python scripts/run_holdout_confirmation_campaign.py \
  --task-manifest artifacts/workshop2026/task_selection_manifest.json \
  --candidate-manifest artifacts/workshop2026/candidate_manifest.json \
  --kernelbench-dir /workspace/KernelBench \
  --campaign-validity artifacts/workshop2026/campaign_validity.json \
  --confirmation-wave wave1
```

Wave 2 is a separate invocation after the frozen integrity lock permits it. The
analysis requires exactly seven process records per task and never replaces a
missing process. RQ2 does not reuse the three-candidate holdout pool: it confirms
all 20 deterministic candidates per controlled fused8 task with
`scripts/run_multiplicity_campaign.py`.

The checked-in campaign is complete. Its formal gate is `PASS`; the holdout
ledger contains 249 entries and the multiplicity ledger contains 67 entries.
`scripts/make_workshop2026_results_figure.py` regenerates the final three-panel
figure from these artifacts. The primary result is 0 screening and 0 confirmed
wins above the 2% margin among 10 valid task winners; see
`reports/workshop2026_corrected_results.md` for traced summary values.

The deterministic source files omitted from one intermediate artifact import
can be restored only when their regenerated source and metadata bytes match the
pre-timing manifest:

```bash
python scripts/restore_frozen_multiplicity_candidates.py
```

The same-GPU easy-grid control and the single-candidate compiler-rung check use
only those frozen candidates:

```bash
python scripts/run_multiplicity_campaign.py \
  --protocol configs/workshop2026_multiplicity_protocol.yaml \
  --candidate-manifest artifacts/workshop2026/multiplicity/candidate_manifest.json \
  --output-dir artifacts/workshop2026/multiplicity_same_gpu_a4500 \
  --max-gpu-hours 2.0
python scripts/analyze_same_gpu_rq2_control.py

python scripts/run_compiler_rung_confirmation.py \
  --kernelbench-dir /content/KernelBench \
  --output-dir artifacts/workshop2026/compiler_confirmation_a4500 \
  --processes 7
```

These controls add separate evidence tables; they do not overwrite the primary
T4 holdout or near-threshold A4500 campaign.

The calibrated near-threshold stress test is a separate deterministic RQ2
campaign on an RTX A4500. From a clean checkout with the historical easy-grid
artifacts present, its frozen command sequence is:

```bash
python scripts/freeze_near_threshold_candidates.py \
  --protocol configs/workshop2026_near_threshold_multiplicity_v3_protocol.yaml \
  --output-root artifacts/workshop2026/near_threshold_multiplicity_v3/candidate_pool
python scripts/run_near_threshold_multiplicity_campaign.py \
  --stage calibration \
  --protocol configs/workshop2026_near_threshold_multiplicity_v3_protocol.yaml \
  --candidate-pool-manifest artifacts/workshop2026/near_threshold_multiplicity_v3/candidate_pool_manifest.json \
  --selected-manifest artifacts/workshop2026/near_threshold_multiplicity_v3/selected_candidate_manifest.json \
  --output-dir artifacts/workshop2026/near_threshold_multiplicity_v3/campaign \
  --max-gpu-hours 4.7
python scripts/run_near_threshold_multiplicity_campaign.py \
  --stage screening \
  --protocol configs/workshop2026_near_threshold_multiplicity_v3_protocol.yaml \
  --candidate-pool-manifest artifacts/workshop2026/near_threshold_multiplicity_v3/candidate_pool_manifest.json \
  --selected-manifest artifacts/workshop2026/near_threshold_multiplicity_v3/selected_candidate_manifest.json \
  --output-dir artifacts/workshop2026/near_threshold_multiplicity_v3/campaign \
  --max-gpu-hours 4.7
python scripts/run_near_threshold_multiplicity_campaign.py \
  --stage confirmation --wait-for-separation \
  --protocol configs/workshop2026_near_threshold_multiplicity_v3_protocol.yaml \
  --candidate-pool-manifest artifacts/workshop2026/near_threshold_multiplicity_v3/candidate_pool_manifest.json \
  --selected-manifest artifacts/workshop2026/near_threshold_multiplicity_v3/selected_candidate_manifest.json \
  --output-dir artifacts/workshop2026/near_threshold_multiplicity_v3/campaign \
  --max-gpu-hours 4.7
python scripts/analyze_near_threshold_campaign.py
```

Calibration is excluded from primary estimates. The confirmation runner enforces
the recorded 30-minute separation and requires exactly seven fresh processes for
every selected candidate. The accepted v3 ledger contains 94 files; the derived
summary records three apparent and two confirmed full-budget wins. The preceding
v1 and v2 calibration grids are design provenance only and produced no primary
estimates.

The historical command below is retained for provenance and must not be reused
as a corrected result. API keys must be environment variables only; never
commit them. No additional candidate generation is required to reproduce the
paper from the frozen manifest.

```bash
export GEMINI_API_KEY=<key>
python -m openkernelforge.cli kernelbench-l1-check \
  --config configs/kernelbench_l1_20task_gemini_rigorous.yaml \
  --kernelbench-dir /workspace/KernelBench
unset GEMINI_API_KEY
```

Statically audit the preserved historical candidates without executing them:

```bash
python scripts/audit_historical_kernelbench_candidates.py
```

Any new repair pass must point to failures from the corrected candidate run. The checked-in `kernelbench_l1_20task_gemini_repair1.yaml` is historical provenance because it names the affected parent run; do not reuse it for corrected claims.

The historical original-vs-repair comparison is recorded at:

```text
runs/kernelbench_gemini_repair1_comparison.md
```

Historical evidence status: the affected evaluator recorded 3/20 one-shot verifications and 1/8 repair verifications. These are not current correctness or performance results. The corrected 48-task campaign supersedes them for paper-facing external evidence.

The paper appendix includes two concrete candidate examples from imported
artifacts: a historically accepted `CrossEntropyLoss` source and a failed
`matmul_4d` Triton compile-error candidate. The source files are preserved
under `artifacts/runpod_imports/runs/20260520_202314/candidates/`.

## Notes

- GPU is required for real Triton correctness/performance claims.
- Rigorous fused8 runs used 30 warmup iterations, 120 measured samples per
  session, and 3 same-process loops. Current code rotates measurement order,
  reports session-level summaries, and materializes compile before runtime
  sampling. Historical KernelBench accounting is not paper evidence.
- Cache flushing is implemented as a 128 MB CUDA write-based cache-state
  perturbation before measured samples. It is not a formal guarantee of full
  L2/cache eviction.
- Imported fused8 summaries use legacy repeatability labels because full
  session-label inputs are not preserved locally. Historical KernelBench files
  contain rigorous-label fields, but those labels are provisional because the
  adapter was invalid.
- The fused8 suite uses one primary shape regime centered on `[4096, 1024]`.
  It characterizes that controlled regime rather than scaling behavior.
- API keys must be environment variables only.
- `runs/` and `datasets/` should be reviewed before using them for training.
- Historical KernelBench rows remain evaluator-audit artifacts. The corrected
  workshop campaign under `artifacts/workshop2026/` is the supported external
  evidence and is reported separately.

## Paper Build

The Overleaf-ready source package is `paper/overleaf/`.

```bash
python scripts/make_paper_figures.py
python scripts/build_paper_assets.py
python scripts/build_paper_pdf.py
```

For the formal workshop paper, run
`python scripts/build_workshop2026_paper.py --submission-ready`; the output is
`paper/workshop2026/openkernelforge_workshop2026.pdf`. The longer Overleaf
package remains useful as an audit appendix. Historical KernelBench rows are
explicitly provisional; the corrected workshop result is sourced only from the
checksummed `artifacts/workshop2026/` campaign.

The Overleaf package is self-contained and includes `paper/overleaf/OVERLEAF_README.md`.
