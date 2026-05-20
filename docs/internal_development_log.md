# OpenKernelForge Internal Development Log

This document preserves the iterative implementation history that used to live in public-facing documentation. It is intentionally internal. The README and paper files should present the project by problem, method, experiments, limitations, and reproducibility rather than by sprint numbering.

## Harness Foundation

- Added task abstractions, simple PyTorch reference tasks, deterministic input generation, verifier, benchmarker, dummy agent, smoke config, JSONL logging, and summary output.
- Added CPU-only-compatible tests for the verifier, benchmarker, runner, and smoke path.

## Model Interface And Repair Loop

- Added provider-agnostic model backends, deterministic fake backend, prompt building, code extraction, and iterative repair prompts.
- Added prompt/response/candidate artifact logging for each attempt.
- Added candidate-level JSONL records and fake repair tests.

## OpenAI-Compatible Backends And Sampling

- Added OpenAI-compatible chat completion backend, backend factory, multi-candidate sampling, selected-best metadata, and config inspection.
- Added local server support for vLLM/LM Studio/llama.cpp-style endpoints.

## Real Baseline And Policy Checks

- Added backend health check, anti-fallback candidate policy, real-baseline configs, run comparison script, and improved summaries.
- Added static rejection for obvious PyTorch fallback code when fallback mode is disabled.

## Dataset Export And Analysis

- Added failure taxonomy, analyze-run command, dataset export and validation, repair-pair extraction, optimization-pair extraction, and curated fused8 dataset logic.

## CUDA/Triton Environment Awareness

- Added structured environment probe, environment-aware metadata, missing-CUDA/missing-Triton failure classifications, GPU configs, and GPU runner.

## Three-Task Sandbox Findings

- Ran initial CUDA/Triton 3-task baselines.
- Correctness became reliable, but speedups on `vector_add`, `relu`, and `bias_relu` were mostly slow.
- Added deterministic templates, template-copy prompting, invalid variant filtering, and repeatability.
- Final conclusion: simple standalone elementwise tasks were poor targets; move to fused workloads.

## Internal Fused8 Benchmark

- Added fused tasks: `bias_relu`, `sigmoid_mul`, `add_relu`, `residual_add_relu`, `bias_gelu`, `row_sum`, `layernorm_small`, and `rmsnorm_small`.
- Added fused deterministic templates and fused8 reports.
- Deterministic fused templates produced repeat-stable wins on `residual_add_relu`, `bias_gelu`, and `rmsnorm_small`.
- Gemini/OpenAI cheap baselines generated correct fused kernels reliably but did not dominate deterministic templates.
- Qwen 7B local zero-shot was weak. Qwen 14B was not tested because the pod ran out of disk/cache while downloading the model.

## Report And Artifact Packaging

- Added technical report, reproducibility guide, artifact index, artifact preservation plan, package/import scripts, research package validation, and release checklist.
- Public documentation was rewritten around the paper thesis: repeatability-aware CUDA benchmarking prevents false speedup claims.
