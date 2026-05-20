# OpenKernelForge

OpenKernelForge is an open research harness for verifier-guided GPU kernel generation and optimization.

The long-term goal is an open-model, verifier-guided, profiler-aware Triton/CUDA kernel generation agent that can compete on KernelBench/TritonBench-style workloads. This repository is intentionally building the research foundation first: task definitions, candidate execution, correctness verification, benchmarking, repeatability checks, dataset curation, and auditable run artifacts.

Current status: OpenKernelForge has a working internal fused8 benchmark, deterministic Triton template baselines, repeatability-aware evaluation, curated dataset export, local/OpenAI-compatible model support, and technical reports under `reports/`. These are internal fused8 results, not KernelBench results and not a SOTA claim.

## Research Snapshot

OpenKernelForge has moved from a three-task harness sanity check into a fused-kernel benchmark that better reflects where Triton can help. The main finding so far is conservative but useful: LLMs can generate correct fused Triton kernels, but stable speedup is harder than correctness. Deterministic templates remain a strong baseline, template guidance does not automatically improve performance, and repeatability is required before trusting single-run wins.

Key internal fused8 observations:

- Deterministic fused8 templates verified `2076/2076` candidates and produced repeat-stable wins for `residual_add_relu`, `bias_gelu`, and `rmsnorm_small`.
- Gemini produced correct fused kernels reliably and was competitive, but not dominant.
- OpenAI mini and GPT-5.5 cheap runs were correct and competitive under the same small-budget protocol, but did not clearly beat Gemini or deterministic templates.
- Qwen 7B was weak zero-shot on fused8; Qwen 14B was not evaluated because the vLLM pod ran out of disk/cache during model download.
- The curated fused8 dataset separates repeat-stable fast rows from single-run-only, promising, optimization-pair, and rejected/unstable rows.

What this is not:

- Not a KernelBench submission.
- Not a SOTA claim.
- Not a trained model.
- Not an RL system.
- Not a replacement for profiling; Nsight-style profiling is still future work.

## Why This Matters

Kernel generation is a real frontier-lab bottleneck. Strong systems need more than a prompt and a compiler error. They need benchmark-shaped tasks, deterministic verification, hardware-aware timing, repair signals, and enough run metadata to train or evaluate future open models.

OpenKernelForge is designed around KernelBench/TritonBench-style evaluation, while keeping the first MVP usable on CPU-only machines.

## What Currently Works

- Simple task interface with eight built-in PyTorch tasks.
- Internal fused8 task set with deterministic input generation, tolerances, shape metadata, and prompt hints.
- Correctness verifier with multi-seed checks, shape and dtype checks, NaN/Inf detection, and max error reporting.
- Benchmarker for PyTorch eager, candidate code, and optional `torch.compile`.
- Dummy agent that emits torch fallback candidates, plus small Triton candidates for `vector_add` and `relu` when CUDA and Triton are available.
- Pluggable LLM agent interface with deterministic fake backend for offline tests.
- Real OpenAI-compatible backend for local servers such as vLLM, LM Studio, llama.cpp, or custom compatible endpoints.
- Multi-candidate sampling per attempt with best-candidate selection from correctness and benchmark signals.
- Iterative repair loop that feeds verifier failures back into the model backend.
- Prompt, raw response, candidate, error log, and attempt-level JSONL artifacts.
- Smoke run that writes artifacts under `runs/<timestamp>/`.
- JSONL result logging and Markdown summary generation.
- Deterministic Triton template baselines, template autotuning, variant validation, repeatability reports, and fused8 dataset curation.
- Research packaging with `reports/openkernelforge_technical_report.md`, reproducibility instructions, CSV table sources, and artifact index generation.
- Preliminary KernelBench path discovery that fails gracefully when KernelBench is absent.

## What Does Not Work Yet

- Full KernelBench task conversion and evaluation are not implemented yet.
- Transformers/vLLM in-process backends are placeholders; local models are served through an OpenAI-compatible server.
- No fine-tuning yet.
- No RL yet.
- No Nsight profiler yet.
- No SOTA claims.
- KernelBench importer is preliminary and does not convert full KernelBench tasks into `KernelTask` objects yet.

## Install

Use an environment with PyTorch installed. Triton is optional for this MVP.

```bash
pip install -e .
```

For development tests:

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m openkernelforge.cli smoke
python -m openkernelforge.cli run --config configs/smoke.yaml
python -m openkernelforge.cli run --config configs/smoke.yaml --agent llm --backend fake
python -m openkernelforge.cli run --config configs/fake_multi_candidate.yaml
python -m openkernelforge.cli show-config --config configs/local_openai_compatible.yaml
python -m openkernelforge.cli summarize --run-dir runs/<timestamp>
python -m openkernelforge.cli inspect-run --run-dir runs/<timestamp>
```

A run writes:

```text
runs/
  <timestamp>/
    config.yaml
    results.jsonl
    summary.md
    prompts/
      <task_id>/
        candidate_000_prompt.txt
    responses/
      <task_id>/
        candidate_000_response.txt
    candidates/
      <task_id>/
        candidate_000.py
    logs/
      <task_id>/
        candidate_000.err.txt
```

Error logs are only created when a candidate fails to load, verify, or benchmark.

## Research Reports And Reproducibility

Build the Phase 14 research package without GPU or API access:

```bash
python scripts/build_phase14_report.py
python scripts/check_research_artifacts.py
```

This creates:

- `reports/openkernelforge_technical_report.md`
- `reports/reproducibility.md`
- `reports/artifact_index.md`
- CSV table sources under `reports/tables/`
- optional PNG figures under `reports/figures/` when matplotlib is available

The report separates observed artifact-backed results from provided run summaries when the RunPod artifacts are not present in the current workspace.

## Artifact Preservation

Phase 15 adds a reproducible path for preserving the real RunPod fused8 artifacts without copying secrets, model weights, or Hugging Face caches.

On RunPod:

```bash
cd /workspace/openkernelforge
python scripts/package_runpod_artifacts.py \
  --out openkernelforge_fused8_artifacts.tar.gz
```

Locally:

```bash
python scripts/import_runpod_artifacts.py \
  --archive openkernelforge_fused8_artifacts.tar.gz \
  --out artifacts/

python scripts/validate_research_package.py
python scripts/update_artifact_index.py
```

See `reports/artifact_preservation_plan.md` for the exact source paths and expected files.

## Current Artifact Status

This checkout includes the source code, reports, generated CSV tables, and summary metrics. The full fused8 RunPod artifacts and `datasets/fused8_curated_v1` are not present in this local workspace unless imported under `artifacts/`.

`reports/artifact_index.md` records whether each real run artifact is present locally, summarized only, missing, or optional missing. Missing artifacts are not fabricated.

## LLM Agent Architecture

The LLM path is provider-agnostic:

- `ModelBackend.generate(prompt, system=None, **kwargs)` is the minimal backend contract.
- `FakeBackend` is deterministic, offline, and used by tests.
- `LLMAgent` builds task prompts, calls a backend, extracts Python code, and prepares repair prompts.
- `code_extract.py` handles raw Python, fenced Markdown code blocks, and responses with prose around code.
- `repair.py` converts verifier failures into compact repair feedback.

The default configs still use `agent.type: dummy` so the original smoke behavior stays stable. To exercise the LLM loop without API keys:

```bash
python -m openkernelforge.cli run --config configs/smoke.yaml --agent llm --backend fake
```

The fake backend can also be configured in tests with `fake_mode: broken_then_fixed` to force one failed attempt followed by a repaired candidate.

## Repair Loop

For each task, the LLM runner:

1. Builds a task prompt from task metadata and reference source.
2. Stores the prompt and raw model response.
3. Extracts candidate Python code and saves it under `candidates/`.
4. Imports `forward(*args)` and runs the verifier.
5. If verification fails, builds a repair prompt with the previous code, structured verifier output, shape/dtype details, errors, and max error values.
6. Repeats up to `agent.max_attempts`.
7. Benchmarks correct candidates and marks the selected best candidate.

`results.jsonl` now contains candidate-level records plus a final task-level summary. Candidate records include attempt index, candidate index, backend, model, prompt path, response path, candidate path, verification summary, benchmark summary when benchmarked, `selected_best`, failure reason, and creation time.

## Multi-Candidate Sampling

Set `agent.candidates_per_attempt` above 1 to sample multiple candidates before deciding whether to repair:

```bash
python -m openkernelforge.cli run --config configs/fake_multi_candidate.yaml
```

The runner verifies every candidate, benchmarks correct candidates, then selects the best candidate by correctness, speedup vs `torch.compile` when available, speedup vs PyTorch eager, and median runtime. By default, `stop_after_first_correct: true` stops the repair loop once an attempt contains a correct candidate.

## Local OpenAI-Compatible Servers

Use `backend: openai_compatible` for any server exposing `/v1/chat/completions`:

```bash
python -m openkernelforge.cli run \
  --config configs/local_openai_compatible.yaml \
  --agent llm \
  --backend openai_compatible \
  --model qwen3-coder \
  --base-url http://localhost:8000/v1
```

Example vLLM command, shown only as a starting point:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

Other compatible servers, including LM Studio and llama.cpp, can be used if they expose the same chat completions endpoint. Real model quality depends on the served model, prompt behavior, hardware, and sampling settings; this project does not claim benchmark success from the fake backend.

API keys are optional for local servers. If configured, use environment variables such as `OPENAI_API_KEY`; saved run configs are redacted so secrets are not written to artifacts.

The OpenAI-compatible backend uses the optional `requests` package. If your environment does not already include it, install it before running against a real server.

## Running Local Open Models With vLLM

For Phase 13, local/open models use the same OpenAI-compatible `/v1/chat/completions` backend as the remote API runs, but the server is expected to run on the GPU machine. This avoids paid API calls and keeps verification local.

Start a vLLM OpenAI-compatible server, adjusting the model for GPU memory:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <model_name_or_path> \
  --host 0.0.0.0 \
  --port 8000
```

Candidate starting points:

- `Qwen/Qwen2.5-Coder-7B-Instruct`
- `Qwen/Qwen2.5-Coder-14B-Instruct`
- a DeepSeek coder model available in your environment
- a Nemotron coder/reasoning model available in your environment

Check the local server without running a benchmark:

```bash
python scripts/check_local_model_server.py
```

Run the cheap fused8 Qwen baseline, matching the OpenAI mini/GPT-5.5 smoke budget of one candidate per task:

```bash
python scripts/run_local_model_fused8.py \
  --config configs/qwen_fused8_gpu_baseline_cheap.yaml \
  --out-name qwen_fused8_cheap
```

The cheap local configs do not set an API key by default. If your local server requires one, set the config's `api_key_env` and provide that environment variable. Compare all available fused8 runs with:

```bash
python scripts/compare_all_fused8_models.py \
  --out runs/fused8_all_model_comparison.md
```

These are internal fused8 comparisons only, not KernelBench results and not SOTA claims.

## Running The First Real Model Baseline

OpenKernelForge supports any local or remote server that implements the OpenAI-compatible chat completions API. Three practical paths:

1. vLLM OpenAI-compatible server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <model_name_or_path> \
  --host 0.0.0.0 \
  --port 8000
```

2. LM Studio local server:

Start LM Studio's OpenAI-compatible local server and use:

```text
http://localhost:1234/v1
```

as the `base_url`.

3. Any OpenAI-compatible endpoint:

```bash
python -m openkernelforge.cli run \
  --config configs/real_baseline_3tasks.yaml \
  --backend openai_compatible \
  --base-url http://localhost:8000/v1 \
  --model <model>
```

Google Gemini can also be used through its OpenAI-compatible endpoint. The included Gemini config targets the public `gemini-3.1-flash-lite` model and reads the key from `GEMINI_API_KEY`:

```bash
export GEMINI_API_KEY=<your-key>
python -m openkernelforge.cli check-backend --config configs/gemini_3_1_flash_lite_baseline_3tasks.yaml
python -m openkernelforge.cli run --config configs/gemini_3_1_flash_lite_baseline_3tasks.yaml
```

Before running, check connectivity:

```bash
python -m openkernelforge.cli check-backend --config configs/real_baseline_3tasks.yaml
```

The one-command real baseline protocol is:

```bash
python scripts/run_real_baseline_3tasks.py
```

It checks the backend, runs the real baseline only if the backend is reachable, then analyzes, exports, and validates the dataset. It never creates fake real-model results.

For a harness comparison that works even without a real server:

```bash
python scripts/run_baseline_comparison_3tasks.py
```

This runs dummy and fake baselines, runs the real baseline only if the backend is available, and prints a comparison table for available runs.

## Running The Gemini Baseline On A CUDA Machine

CPU-only runs are useful for checking model generation, prompt logging, extraction, policy behavior, and dataset plumbing. They should not be used for true Triton correctness or performance claims. For that, run on a machine with CUDA and Triton execution working.

Check the local execution environment first:

```bash
python -m openkernelforge.cli env-check
python -m openkernelforge.cli env-check --out environment_probe.json
```

The viability status is one of:

- `CPU_ONLY`: no CUDA; Triton kernels cannot be verified or benchmarked.
- `CUDA_NO_TRITON`: CUDA exists but Triton is not importable.
- `TRITON_IMPORT_ONLY`: CUDA and Triton import, but a tiny Triton kernel fails.
- `TRITON_EXECUTION_OK`: suitable for Triton verification and benchmarking.
- `UNKNOWN_BROKEN`: probing failed before a clearer category could be assigned.

Every run now writes `runs/<timestamp>/environment_probe.json`, and summaries/reviews separate local environment failures from model failures.

Workflow A: Gemini API generation plus local GPU verification:

```bash
export GEMINI_API_KEY=<your-key>
python -m openkernelforge.cli env-check
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/gemini_3_1_flash_lite_baseline_3tasks_gpu.yaml \
  --out-name gemini_gpu_3tasks_v1
```

This uses Gemini for candidate generation and the local CUDA/Triton machine only for verification and benchmarking. It does not require serving a local model.

Workflow B: local open model through vLLM plus local GPU verification:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <model_name_or_path> \
  --host 0.0.0.0 \
  --port 8000
```

Then in another shell:

```bash
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/real_baseline_3tasks_gpu.yaml \
  --out-name local_model_gpu_3tasks_v1
```

The GPU runner refuses to continue unless CUDA, Triton import, and a tiny Triton kernel all work. Fake and dummy runs remain harness checks only.

## GPU Baseline V2 Prompting

The first RunPod GPU baseline with Gemini 3.1 Flash-Lite verified correctness for all three tasks, but the selected kernels were slower than PyTorch eager. That result is promising for correctness and useful for SFT, repair, and optimization data, but it is not a performance win and not a SOTA claim.

Phase 9.7 adds:

- `v2_task_skeletons`: task-specific Triton skeleton hints for `vector_add`, `relu`, and `bias_relu`.
- `v3_cuda_repair`: CUDA-aware repair prompts for correct-but-slow candidates and Triton compile errors.
- `debrief-gpu-run`: static GPU candidate debrief reports.
- GPU v1/v2 comparison tooling.

Run the v2 prompt baseline:

```bash
export GEMINI_API_KEY=<key>
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/gemini_3_1_flash_lite_baseline_3tasks_gpu_v2.yaml \
  --out-name gemini_gpu_3tasks_v2
```

Run fastsearch for more optimization candidates:

```bash
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/gemini_3_1_flash_lite_baseline_3tasks_gpu_v2_fastsearch.yaml \
  --out-name gemini_gpu_3tasks_v2_fastsearch
```

Debrief a GPU run:

```bash
python -m openkernelforge.cli debrief-gpu-run --run-dir runs/<gpu_run>
```

Compare v1 and v2:

```bash
python scripts/compare_gpu_v1_v2.py \
  --v1 runs/<v1> \
  --v2 runs/<v2>
```

If `--v2` is omitted or missing, the comparison script prints the exact v2 command to run. Correct-but-slow examples should be treated as optimization data, not high-quality performance targets. No fine-tuning or RL is part of this phase.

## First Real Model Baseline

This protocol is for the first real-model baseline on three simple tasks: `vector_add`, `relu`, and `bias_relu`. It is not a SOTA claim. The goal is to collect the first real prompts, responses, policy rejections, verifier failures, and benchmark numbers.

1. Start a local OpenAI-compatible server. Example vLLM-style command, shown only as a placeholder:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <your-model> \
  --host 0.0.0.0 \
  --port 8000
```

2. Check backend health:

```bash
python -m openkernelforge.cli check-backend --config configs/real_baseline_3tasks.yaml
```

3. Run the dummy harness baseline:

```bash
python -m openkernelforge.cli run --config configs/dummy_baseline_3tasks.yaml
```

4. Run the fake LLM harness baseline:

```bash
python -m openkernelforge.cli run --config configs/fake_baseline_3tasks.yaml
```

5. Run the real model baseline:

```bash
python -m openkernelforge.cli run --config configs/real_baseline_3tasks.yaml
```

6. Compare runs:

```bash
python scripts/compare_runs.py runs/<dummy> runs/<fake> runs/<real>
```

The dummy and fake baselines are harness sanity checks, not real model benchmarks. In the real baseline config, `allow_torch_fallback: false`, so obvious PyTorch fallback responses such as direct `x + y` or `torch.relu(x)` are rejected by the policy checker before verification. Real model quality depends on the model served, decoding settings, and hardware.

After a real run, generate a review report:

```bash
python -m openkernelforge.cli review-real-run --run-dir runs/<real_run>
```

The report writes `real_run_review.md` and summarizes per-task outcomes, common model behavior, prompt weaknesses, dataset usefulness, and a human review checklist.

## Candidate Policy

Before import and verification, OpenKernelForge runs a lightweight static policy check. It requires `forward`, blocks obvious reference/task imports, blocks suspicious calls such as `reference` or `ref_forward`, and rejects clear PyTorch fallbacks when torch fallback mode is disabled, including fallback paths hidden behind `try: import triton ... except ImportError`. The checker is intentionally conservative: uncertain cases become warnings rather than hard failures. Policy results are stored in candidate-level JSONL fields.

## Phase 8: From Real Model Runs To Training Data

OpenKernelForge can now analyze run artifacts and export inspectable JSONL datasets for future SFT, repair, optimization, and rejected/failure analysis. This does not train a model and does not make a run training-quality by itself.

Recommended flow:

1. Start a local OpenAI-compatible server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <your-model> \
  --host 0.0.0.0 \
  --port 8000
```

2. Check backend:

```bash
python -m openkernelforge.cli check-backend --config configs/real_baseline_3tasks.yaml
```

3. Run real baseline:

```bash
python -m openkernelforge.cli run --config configs/real_baseline_3tasks.yaml
```

4. Analyze:

```bash
python -m openkernelforge.cli analyze-run --run-dir runs/<real_run>
```

5. Export dataset:

```bash
python -m openkernelforge.cli export-dataset \
  --run-dir runs/<real_run> \
  --out-dir datasets/real_baseline_3tasks_v1
```

6. Validate dataset:

```bash
python -m openkernelforge.cli validate-dataset --dataset-dir datasets/real_baseline_3tasks_v1
```

Exported datasets contain:

- `sft_raw.jsonl`: policy-passing, verified candidates for manual review as raw SFT targets.
- `repair.jsonl`: failed-to-later-correct pairs with verifier or policy feedback.
- `optimization.jsonl`: slower-correct to faster-correct pairs when benchmark data supports the comparison.
- `rejected.jsonl`: rejected or failed candidates for analysis, not direct training targets.
- `manifest.json`: counts, failure taxonomy, run kind, and warnings.

Fake and dummy runs are useful for testing the pipeline, but they are marked harness-only and should not be treated as real model training data. Real-model exports should be reviewed before fine-tuning. A correct kernel is not automatically high quality if it is slower than baseline; repair and optimization pairs are likely the most valuable future training data.

## Phase 9.8: Performance Search

The first CUDA/Triton Gemini baseline produced mostly correct but slow kernels. Performance search is a separate post-correctness stage: after a candidate passes policy, verification, and benchmarking but misses the speed target, OpenKernelForge asks the backend for optimized variants using the candidate code, benchmark feedback, and static performance heuristics.

Run the narrow performance-search baseline on a CUDA/Triton machine:

```bash
export GEMINI_API_KEY=<key>
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/gemini_3_1_flash_lite_3tasks_gpu_perfsearch.yaml \
  --out-name gemini_gpu_3tasks_perfsearch
```

For a wider data-collection search:

```bash
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/gemini_3_1_flash_lite_3tasks_gpu_perfsearch_wide.yaml \
  --out-name gemini_gpu_3tasks_perfsearch_wide
```

Compare against the v2 fastsearch run:

```bash
python scripts/compare_perfsearch.py \
  --baseline runs/<v2_fastsearch_run> \
  --perfsearch runs/<perfsearch_run>
```

Performance-search records include `generation_stage`, `search_round`, `parent_candidate_path`, `parent_speedup_vs_eager`, `improved_over_parent`, `best_initial_speedup_vs_eager`, `best_final_speedup_vs_eager`, and `target_reached`. The run also writes `performance_search_report.md`. These results are for optimization research and dataset collection; they are not SOTA claims.

## Phase 9.9: Deterministic Template Baselines

OpenKernelForge includes deterministic Triton template baselines for `vector_add`, `relu`, and `bias_relu`. The template agent does not call a model. It generates a fixed autotune grid over `BLOCK_SIZE`, `num_warps`, and wrapper contiguity policy, then sends every candidate through the same policy, verifier, benchmarker, JSONL logging, reports, and dataset export pipeline.

Run template autotune on a CUDA/Triton machine:

```bash
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/template_3tasks_gpu_autotune.yaml \
  --out-name template_3tasks_gpu_autotune
```

Generate or refresh the template report:

```bash
python -m openkernelforge.cli template-report --run-dir runs/<template_run>
```

Compare an LLM run against templates:

```bash
python scripts/compare_llm_vs_templates.py \
  --llm runs/<llm_run> \
  --template runs/<template_run>
```

Template candidate records are labeled with `generation_stage=template_baseline`, `template_family`, `template_id`, `block_size`, `num_warps`, and `contiguous_policy`. Dataset rows include `source_type=template` for template-generated code. These baselines are intended to establish a simple deterministic performance floor before more LLM optimization work.

Template-guided LLM performance search is configured separately in:

```bash
configs/gemini_3_1_flash_lite_3tasks_gpu_template_guided_perfsearch.yaml
```

Set `agent.performance_search.template_run_dir` to a completed template run before using it. If the template run is missing, OpenKernelForge fails clearly rather than silently dropping the template context.

## Phase 10.0: Profiler-Lite And Strict Template Copy

Profiler-lite is a benchmark/statics report, not Nsight and not a hardware profiler. It reads saved benchmark distributions and candidate source code, then flags source patterns such as extra torch ops in `forward`, added `.contiguous()` calls, missing `@triton.jit`, missing `BLOCK_SIZE: tl.constexpr`, multiple kernel launches, missing bias modulo indexing, and try/except fallback wrappers.

Generate the report for any existing run:

```bash
python -m openkernelforge.cli profiler-lite --run-dir runs/<run_id>
```

The strict template-copy mode asks a model to copy/adapt the best known template with controlled parameter changes. It is intentionally constrained: preserve the template structure, keep one Triton kernel launch, keep the same grid and indexing logic, do not add fallback branches, and do not add high-level torch ops such as `torch.relu`, `torch.maximum`, `torch.add`, `torch.matmul`, or `torch.sigmoid`.

Before running template-copy, run deterministic template autotune and set:

```yaml
agent:
  performance_search:
    mode: template_copy
    include_best_template_context: true
    template_run_dir: runs/<template_run>
```

Run the narrow template-copy baseline on a CUDA/Triton machine:

```bash
export GEMINI_API_KEY=<key>
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/gemini_3_1_flash_lite_3tasks_gpu_template_copy.yaml \
  --out-name gemini_gpu_3tasks_template_copy
```

For a wider parameter sweep:

```bash
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/gemini_3_1_flash_lite_3tasks_gpu_template_copy_wide.yaml \
  --out-name gemini_gpu_3tasks_template_copy_wide
```

Compare template, template-guided, and template-copy runs:

```bash
python scripts/compare_template_copy.py \
  --template runs/<template_run> \
  --template-guided runs/<template_guided_run> \
  --template-copy runs/<template_copy_run>
```

Template-copy candidate records include `generation_stage=template_copy`, `template_source_path`, `copied_from_template_id`, requested parameter fields, `preserved_template_structure_score`, preservation warnings, fallback flags, and benchmark delta versus the source template. Dataset exports mark these rows with `source_type=template_copy` and export `template_copy_optimization` rows separately.

## Phase 10.1: Expanded Template Sweeps

The deterministic template agent supports wider, capped template sweeps over `BLOCK_SIZE`, `num_warps`, `num_stages`, contiguous policy, output allocation policy, and optional shape-aware modes. The grid is capped with deterministic ordering by `max_variants_per_task` so larger sweeps stay auditable.

Run the expanded deterministic sweeps on a CUDA/Triton machine:

```bash
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/template_3tasks_gpu_autotune_wide.yaml \
  --out-name template_3tasks_gpu_autotune_wide

python scripts/run_gpu_baseline_3tasks.py \
  --config configs/template_3tasks_gpu_autotune_shapeaware.yaml \
  --out-name template_3tasks_gpu_autotune_shapeaware
```

For a quick sanity pass:

```bash
python scripts/run_gpu_baseline_3tasks.py \
  --config configs/template_3tasks_gpu_autotune_quick.yaml \
  --out-name template_3tasks_gpu_autotune_quick
```

Template reports now write:

- `template_autotune_report.md`
- `template_leaderboard.csv`
- `template_leaderboard.json`

Compare template sweeps against template-copy-wide:

```bash
python scripts/compare_template_sweeps.py \
  --base-template runs/<old_template_run> \
  --wide-template runs/<wide_template_run> \
  --shapeaware-template runs/<shapeaware_template_run> \
  --template-copy-wide runs/<template_copy_wide_run>
```

## Built-in Tasks

- `vector_add`
- `elementwise_mul`
- `relu`
- `bias_relu`
- `sigmoid_mul`
- `row_sum`
- `layernorm_small`
- `matmul_bias`

Each task includes a PyTorch reference implementation, deterministic input generation, dtype/tolerance metadata, and default benchmark shapes.

## Tests

```bash
pytest -q
```

The test suite is robust to CPU-only environments.

## Next Phases

- Phase 10.1: run strict template-copy on GPU and compare against deterministic templates.
- Phase 10.2: decide whether the bottleneck is model quality, wrapper preservation, or deterministic search breadth.
- Phase 11: curate SFT/optimization data from verified and reviewed runs.
- Phase 12: LoRA fine-tuning.
- Phase 13: execution-feedback RL.

## Notes

OpenKernelForge imports and runs candidate Python files locally. This is suitable for local research in trusted environments, not for executing untrusted code.
