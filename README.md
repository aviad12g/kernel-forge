# OpenKernelForge

OpenKernelForge is an open research harness for evaluating LLM-generated Triton kernels with correctness verification, repeatability-aware benchmarking, deterministic template baselines, and inspectable run artifacts.

The current thesis is deliberately narrow:

> Single-run speedups from LLM-generated GPU kernels are not reliable enough. Repeatability-aware CUDA benchmarking changes conclusions and prevents false wins.

This repository is not a KernelBench submission, not a trained model release, and not a SOTA claim.

## What This Project Is

OpenKernelForge provides the infrastructure needed to study generated GPU kernels without relying on one-off timing results:

- task definitions with PyTorch references and deterministic input generation
- candidate extraction, static policy checks, and sandboxed loading
- correctness verification before benchmarking
- PyTorch eager and optional `torch.compile` comparisons
- candidate-level JSONL logging with prompts, responses, source, errors, and benchmark summaries
- deterministic Triton template baselines
- repeatability reports for top candidates
- curated dataset export for later manual review and training research
- technical reports and artifact-preservation tooling

The repo currently focuses on an internal fused8 benchmark. KernelBench L1 support is planned as a repeatability study, but has not been run yet.

## Why Repeatability Matters

GPU kernel timing is noisy. In this project, some single-run wins disappeared when top candidates were rebenchmarked. That changed the interpretation of the results:

- correctness did not imply speed
- single-run speedups were sometimes unstable
- deterministic templates remained a strong baseline
- template-guided prompting produced useful optimization data but did not automatically improve performance
- repeat-stable wins were much rarer and more informative than single-run wins

The goal is to make these distinctions explicit before claiming an generated kernel is faster.

## System Overview

```text
task metadata
  -> prompt or deterministic template generator
  -> candidate Python source exposing forward(*args)
  -> static policy check
  -> correctness verifier
  -> benchmarker
  -> repeatability report for top candidates
  -> JSONL logs, reports, and curated datasets
```

Supported agent paths include deterministic templates, a fake backend for tests, OpenAI-compatible model servers, and local vLLM-style servers. Tests do not require CUDA, internet, API keys, or a live model server.

## Current Internal Fused8 Results

These are internal fused8 results, not KernelBench.

| Baseline | Candidates | Verified | Median speedup vs eager | Repeat-stable wins | Interpretation |
| --- | ---: | --- | ---: | --- | --- |
| deterministic templates | 2076 | 2076/2076 | 0.862x | `residual_add_relu`, `bias_gelu`, `rmsnorm_small` | strongest overall floor |
| Gemini baseline | 28 | 28/28 | 0.933x | competitive, not dominant | correct fused kernels reliably |
| Gemini template-guided | 34 | 34/34 | 0.798x | `residual_add_relu` | useful optimization data, worse median |
| OpenAI mini cheap | 8 | 8/8 | 0.882x | `residual_add_relu`, `bias_gelu`, `rmsnorm_small` | cheap and competitive |
| GPT-5.5 cheap | 8 | 8/8 | 0.927x | `bias_gelu`, `rmsnorm_small` | not clearly better under cheap protocol |
| Qwen 7B local | 8 | 1/8 effective | 0.002x | none | weak zero-shot |

Qwen 14B was not evaluated because the vLLM pod ran out of disk/cache during model download. That is not a model-quality result.

## Install

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -r requirements.txt
```

## Run Tests

```bash
pytest -q
```

The test suite is intended to pass on CPU-only machines.

## Run Smoke

```bash
python -m openkernelforge.cli smoke
```

The smoke command uses local harness components and does not require CUDA or API keys.

## Environment Check

```bash
python -m openkernelforge.cli env-check
```

Real Triton performance claims require `TRITON_EXECUTION_OK`. CPU-only runs are useful for testing generation, parsing, policy checks, dataset export, and reports, but not for CUDA/Triton benchmark claims.

## Run The Internal Fused8 Template Benchmark

On a CUDA/Triton machine:

```bash
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/template_fused8_gpu_autotune_quick.yaml \
  --out-name template_fused8_gpu_quick
```

For the wider deterministic sweep:

```bash
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/template_fused8_gpu_autotune_wide.yaml \
  --out-name template_fused8_gpu_wide
```

Generate repeatability reports for top candidates:

```bash
python -m openkernelforge.cli repeatability-report \
  --run-dir runs/<run> \
  --top-k 3 \
  --repeats 5
```

## Reports And Reproducibility

The paper-centered report package is under `paper/` and `reports/`.

```bash
python scripts/build_phase14_report.py
python scripts/check_research_artifacts.py
```

Important files:

- `paper/paper.md`
- `paper/methodology.md`
- `paper/experiments.md`
- `paper/benchmarking_methodology_upgrade.md`
- `paper/kernelbench_l1_pilot_plan.md`
- `reports/openkernelforge_technical_report.md`
- `reports/reproducibility.md`
- `reports/artifact_index.md`

## Artifact Preservation

Full RunPod artifacts are not committed to the repo. Preserve them separately:

```bash
cd /workspace/openkernelforge
python scripts/package_runpod_artifacts.py \
  --out openkernelforge_fused8_artifacts.tar.gz
```

Import locally:

```bash
python scripts/import_runpod_artifacts.py \
  --archive openkernelforge_fused8_artifacts.tar.gz \
  --out artifacts/

python scripts/validate_research_package.py
python scripts/update_artifact_index.py
```

The package scripts exclude API keys, `.env` files, model weights, Hugging Face caches, and unsanitized server logs.

## Current Artifact Status

This repository includes source code, reports, tables, and summary metrics. The full fused8 RunPod artifacts and curated dataset are expected to be imported under `artifacts/` when available. `reports/artifact_index.md` records what is present locally versus summarized only.

## Limitations

- Internal fused8 benchmark only; no KernelBench numbers yet.
- Reported runs were done on a limited hardware and task set.
- No Nsight or hardware-counter profiling yet.
- No model training, LoRA, or RL is included.
- Some report tables use provided RunPod summaries when full artifacts are not present locally.
- No SOTA claim.

## Next Work

The next research sprint is a KernelBench L1 repeatability pilot:

- select 20 L1 tasks
- separate correctness, compile failure, runtime failure, and performance failure
- rebenchmark top-k candidates across independent sessions
- report repeat-stable speedup rate and single-run win decay rate

The planned headline is:

> X% of single-run wins fail repeat verification.

`X` is intentionally blank until the study is actually run.

## Security Note

OpenKernelForge imports and executes candidate Python files locally. Use it in trusted research environments only.
