# OpenKernelForge: Repeatability-Aware Evaluation of LLM-Generated Triton Kernels

## Abstract

Large language models can generate syntactically plausible GPU kernels, and with verifier feedback they can often produce correct Triton implementations. Correctness alone, however, is not enough: GPU timing noise and benchmark variance can make single-run speedups misleading. OpenKernelForge is a research harness for evaluating generated Triton kernels with explicit correctness verification, deterministic template baselines, repeatability-aware benchmarking, and artifact-preserving reports. On an internal fused8 benchmark, several model-generated and deterministic candidates achieved single-run speedups, but repeatability checks changed the final interpretation. The central conclusion is that repeat-stable speedup, not a single benchmark measurement, should be the unit of evidence for generated kernel performance.

## Main Claim

Single-run speedups from LLM-generated GPU kernels are not reliable enough. Repeatability-aware CUDA benchmarking changes conclusions and prevents false wins.

## Contributions

- A verifier-guided Triton kernel evaluation harness with candidate-level artifact logging.
- A deterministic template baseline system for comparing LLM outputs against known-simple Triton implementations.
- Repeatability reports for top candidates to separate stable wins from timing artifacts.
- A curated fused8 dataset structure that separates repeat-stable fast rows from single-run-only, promising, optimization-pair, and rejected/unstable rows.
- A reproducibility and artifact-preservation package for local review.

## Scope

The reported results are from an internal fused8 benchmark, not KernelBench. KernelBench L1 repeatability evaluation is planned but has not been run. No SOTA claim is made.

## Key Findings

- LLMs can generate correct fused Triton kernels under a verifier-guided protocol.
- Correctness does not imply repeat-stable speedup.
- Deterministic templates are a strong baseline and should be compared before model-generated kernels are treated as useful.
- Template guidance can improve dataset usefulness without necessarily improving median performance.
- Small local/open models can fail badly zero-shot; this is useful negative evidence but not a training conclusion.
- Rigorous CUDA-event model reruns are configured for Gemini and OpenAI mini with 24 candidates per model, but only the deterministic template run has been rerun rigorously so far.

## Paper Structure

- `related_work.md`: benchmark and kernel-generation context.
- `methodology.md`: task, verification, policy, benchmark, repeatability, and artifact methods.
- `experiments.md`: current internal fused8 experimental results.
- `limitations.md`: constraints and what the current artifact cannot claim.
- `benchmarking_methodology_upgrade.md`: next sprint's benchmark measurement upgrade plan.
- `kernelbench_l1_pilot_plan.md`: planned KernelBench L1 repeatability pilot.
