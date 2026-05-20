# OpenKernelForge: Verifier-Guided Triton Kernel Generation with Template Baselines and Repeatability-Aware Evaluation

## 1. Abstract

OpenKernelForge is an open-source research harness for verifier-guided Triton kernel generation. The current system combines task definitions, model and template candidate generation, static policy checks, correctness verification, benchmarking, repeatability analysis, and dataset export. This report summarizes the first internal fused8 evaluation campaign. The results are intentionally modest: LLMs can generate correct fused Triton kernels, but deterministic templates remain a strong performance floor and repeatability changes the interpretation of many apparent wins. These are internal fused8 results only, not KernelBench results and not a SOTA claim.

## 2. Motivation

Kernel generation agents need more than pass/fail execution. Useful systems need reproducible prompts, raw responses, candidate sources, policy checks, correctness traces, timing data, repeatability, and structured failure labels. OpenKernelForge was built to make those artifacts first-class so later open-model fine-tuning can be grounded in verified evidence rather than isolated examples.

## 3. System Overview

- Task layer: PyTorch references, deterministic inputs, shape metadata, tolerances, and prompt hints.
- Agent layer: dummy, fake, OpenAI-compatible, local vLLM-compatible, and deterministic template agents.
- Harness layer: candidate extraction, static policy checks, sandboxed import, verifier, benchmarker, and JSONL logging.
- Reporting layer: summaries, run analysis, failure taxonomy, repeatability reports, fused8 reports, and dataset curation.
- Artifact layer: prompt files, raw responses, candidate source, logs, environment probes, datasets, and human-readable reports.

## 4. Benchmark Tasks

The project started with a three-task sandbox: `vector_add`, `relu`, and `bias_relu`. That sandbox validated the harness but showed that isolated elementwise tasks are poor standalone performance targets. The project then moved to an internal fused8 benchmark: `bias_relu`, `sigmoid_mul`, `add_relu`, `residual_add_relu`, `bias_gelu`, `row_sum`, `layernorm_small`, and `rmsnorm_small`. These fused workloads are better aligned with Triton launch amortization and realistic kernel-generation behavior.

## 5. Methods

Each candidate must expose `forward(*args)`. Candidates pass through policy checks before verification. Correct candidates are benchmarked against PyTorch eager and, when configured, `torch.compile`. Legacy artifacts use the original wall-clock timing path; the rigorous opt-in path now records CUDA-event timing, warmup and sample counts, median/IQR, coefficient of variation, optional bootstrap intervals, optional cache flushing, independent sessions, and compile/runtime separation where practical. Repeatability is measured by rebenchmarking top candidates multiple times. Dataset export separates repeat-stable fast candidates, single-run-only candidates, promising candidates, optimization pairs, and rejected or unstable candidates.

## 6. Model and Template Baselines

- Deterministic templates sweep block size, warps, stages, allocation policy, contiguity policy, and shape specialization.
- Gemini fused8 baseline used the cheap fused8 protocol and produced correct kernels reliably.
- Gemini template-guided used deterministic template context but did not improve median speed.
- OpenAI mini and GPT-5.5 cheap runs were correct and competitive but did not clearly dominate Gemini or templates.
- Qwen 7B local zero-shot was weak under the cheap fused8 protocol.
- Qwen 14B was not tested because the vLLM pod ran out of disk/cache space during model download.

## 7. Results

Provenance note: the latest RunPod fused8 artifacts are not present in this local checkout, so these tables use the manually provided run summaries for missing artifacts. `reports/artifact_index.md` records which canonical artifacts are absent locally.

### Three-Task Conclusion

| Task | Best single-run speedup | Repeatability outcome | Final conclusion |
| --- | --- | --- | --- |
| vector_add | 0.692 | repeat median 0.483x | poor standalone target; launch overhead dominates |
| relu | 0.812 | repeat median 0.512x | poor standalone target; use only as harness check |
| bias_relu | 1.017 | repeat median 0.705x | single-run win was not stable; useful as fused-task seed |

### Fused8 Deterministic Template Results

These are the current paper-facing deterministic-template numbers from the rigorous CUDA-event run `runs/20260520_155839`. The older 2076-candidate template-wide table remains useful as historical search evidence, but should be labeled legacy timing.

| Task | Best single-run | Repeat median | Stable above eager | Above torch.compile |
| --- | --- | --- | --- | --- |
| bias_relu | 1.029 | 0.976 | no | yes |
| sigmoid_mul | 0.998 | 0.934 | no | yes |
| add_relu | 0.947 | 0.938 | no | yes |
| residual_add_relu | 1.140 | 1.023 | yes | yes |
| bias_gelu | 1.473 | 1.485 | yes | yes |
| row_sum | 0.790 | 0.674 | no | yes |
| layernorm_small | 0.843 | 0.791 | no | yes |
| rmsnorm_small | 1.674 | 1.452 | yes | yes |

### Model Comparison

| Baseline | Candidates | Verified | Median eager speedup | Tasks above eager | Repeat-stable wins | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| deterministic templates rigorous CUDA-event | 160 | 160/160 | 0.945 | 4/8 single-run | residual_add_relu, bias_gelu, rmsnorm_small | current paper-facing deterministic template table; capped rigorous grid |
| deterministic templates legacy timing | 2076 | 2076/2076 | 0.862 | 5/8 single-run | residual_add_relu, bias_gelu, rmsnorm_small | historical wide search; keep as legacy timing |
| Gemini fused8 baseline | 28 | 28/28 | 0.933 | 4/8 single-run | competitive but not final stable winner in provided summary | strong zero-shot correctness and competitive speed |
| Gemini template-guided | 34 | 34/34 | 0.798 | 4/8 single-run | residual_add_relu | useful optimization data; median performance regressed |
| OpenAI mini cheap | 8 | 8/8 | 0.882 | 3/8 single-run | residual_add_relu, bias_gelu, rmsnorm_small | cheap and competitive; not clearly above Gemini |
| GPT-5.5 cheap | 8 | 8/8 | 0.927 | provided summary: not dominant | bias_gelu, rmsnorm_small | correct and usable; not clearly better under cheap protocol |
| Qwen 7B local | 8 | 1/8 effective | 0.002 | 0/8 | none | not competitive zero-shot |

## 8. Repeatability Analysis

Repeatability changed several conclusions. In the three-task sandbox, single-run wins for `bias_relu` did not survive repeat benchmarking. In fused8, deterministic templates produced repeat-stable wins for `residual_add_relu`, `bias_gelu`, and `rmsnorm_small`; the final stable winner for `residual_add_relu` came from Gemini template-guided. Single-run wins remain useful search signals, but they are not sufficient evidence for benchmark claims.

The benchmarker now supports explicit repeatability labels: `REPEAT_STABLE_WIN`, `SINGLE_RUN_ONLY_WIN`, `UNSTABLE`, `BELOW_EAGER`, and `INSUFFICIENT_DATA`. These labels are intentionally conservative. A single fast sample, or even a single-run task winner, is not sufficient to claim a kernel improvement unless independent measurement sessions preserve the win.

Implementation status: CUDA-event timing, optional cache flushing, independent sessions, and richer sample summaries are implemented as an opt-in rigorous benchmark path. CPU tests pass. A RunPod RTX 5090 methodology check completed at `runs/20260520_145721` with CUDA-event timing, cache flushing performed, and three independent sessions. The configured rigorous fused8 run completed at `runs/20260520_155839` with 160/160 verified candidates. The deterministic template table above is now rigorous CUDA-event timing; the LLM/OpenAI/Gemini rows remain legacy timing until rerun. Rigorous Gemini and OpenAI mini configs exist with 24 candidates per model, but their results are not reported until those model runs complete.

### Stable Winners By Task

| Task | Stable winner | Source type | Repeat median | Interpretation |
| --- | --- | --- | --- | --- |
| bias_relu | none confirmed above eager | n/a | 0.976 | single-run win remains below eager on repeat |
| sigmoid_mul | none confirmed above eager | n/a | 0.934 | near-eager but below eager on repeat |
| add_relu | none confirmed above eager | n/a | 0.938 | near-eager but below eager on repeat |
| residual_add_relu | deterministic template | template | 1.023 | repeat-stable deterministic template win under rigorous timing |
| bias_gelu | deterministic template | template | 1.485 | strong repeat-stable deterministic template win |
| row_sum | none confirmed above eager | n/a | 0.674 | below eager under rigorous timing |
| layernorm_small | none confirmed above eager | n/a | 0.791 | below eager under rigorous timing |
| rmsnorm_small | deterministic template | template | 1.452 | strong repeat-stable deterministic template win |

## 9. Dataset Curation

The curated fused8 dataset separates repeat-stable targets from unstable and single-run-only candidates. That separation is deliberate: single-run-only rows can guide analysis and mining, but should not be used as direct SFT targets without review.

| Split | Rows | Intended use | Train now? |
| --- | --- | --- | --- |
| correct_fast_repeat_stable.jsonl | 19 | reviewed SFT candidates after repeatability | not yet |
| correct_fast_single_run.jsonl | 623 | analysis and candidate mining only | no |
| correct_promising.jsonl | 598 | optimization or ranking data after review | no |
| optimization_pairs_template_vs_gemini.jsonl | 5 | optimization training pairs | not yet |
| optimization_pairs_gemini_vs_template.jsonl | 3 | optimization training pairs | not yet |
| rejected_or_unstable.jsonl | 898 | failure analysis and negative examples | no |

## 10. Failure Modes

- Correct but slow kernels were the dominant model failure mode after prompt hardening.
- Template guidance often preserved correctness but added wrapper or structural overhead in earlier runs.
- Non-power-of-two Triton template variants caused compile failures until variant validation rejected them.
- Qwen 7B zero-shot produced mostly invalid or slow candidates.
- Qwen 14B has no model-quality result because serving failed with `No space left on device` during download/cache.

## 11. Discussion

The most important result is not that one model wins. It is that correctness and speed are separable. Once prompts and policy checks made correctness reliable, speed remained hard. Deterministic templates are not just baselines; they are a floor that model outputs must beat. Template context is useful for producing optimization data, but current models do not automatically preserve fast structure.

## 12. Limitations

- Internal fused8 benchmark only; not KernelBench.
- Small task set and a single GPU class for the reported campaign.
- No Nsight or hardware-counter profiling yet.
- No fine-tuning and no RL.
- Some numbers are provided-run summaries when the full RunPod artifact is not present in this workspace.
- No SOTA claim.

## 13. Next Work

The next technical step is not training immediately. The rigorous internal fused8 deterministic-template results are complete enough to support an external validation pilot. The project now includes a KernelBench L1 adapter and baseline-validation command that can load a local KernelBench-style directory, run PyTorch eager and optional `torch.compile` baselines, and write a pilot report. Candidate generation for KernelBench is intentionally deferred until task loading and baseline timing are validated. No KernelBench performance results are claimed yet.

## 14. Reproducibility Appendix

See `reports/reproducibility.md` for exact command flows. The table sources are in `reports/tables/`. Run artifacts that are not present in the local checkout are listed explicitly in `reports/artifact_index.md`.
