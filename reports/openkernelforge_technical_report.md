# OpenKernelForge: Repeatability-Aware Evaluation for LLM-Generated Triton Kernels

## 1. Abstract

Generated GPU kernels need correctness and independently confirmed speed, not merely plausible source code or one favorable timing sample. OpenKernelForge evaluates Triton candidates with policy checks, contract-aware correctness tests, randomized paired CUDA-event timing, fresh-process confirmation, evaluator controls, and checksummed artifacts. In the corrected workshop campaign, 27 of 141 evaluated candidate records passed the full gate, covering 10 of 48 selected KernelBench L1 tasks; none of the 10 frozen winners exceeded eager by the prespecified 2% margin in screening or confirmation. A separate controlled deterministic fused8 study found no multiplicity-driven false promotion for candidate budgets from 1 through 20. The historical KernelBench pilot remains an evaluator-audit artifact and is excluded from these corrected results.

## 2. Motivation

Kernel generation agents need more than pass/fail execution. Useful systems need reproducible prompts, raw responses, candidate sources, policy checks, correctness traces, timing data, repeatability, and structured failure labels. OpenKernelForge was built to make those artifacts first-class so generated-kernel claims are grounded in repeatable evidence rather than isolated fast samples.

## 3. System Overview

- Task layer: PyTorch references, deterministic inputs, shape metadata, tolerances, and prompt hints.
- Agent layer: dummy, fake, OpenAI-compatible, local vLLM-compatible, and deterministic template agents.
- Harness layer: candidate extraction, conservative AST policy checks, trusted in-process loading, verifier, benchmarker, and JSONL logging.
- Reporting layer: summaries, run analysis, failure taxonomy, repeatability reports, fused8 reports, and dataset curation.
- Artifact layer: prompt files, raw responses, candidate source, logs, environment probes, datasets, and human-readable reports.

## 4. Benchmark Tasks

The project started with a three-task sandbox: `vector_add`, `relu`, and `bias_relu`. That sandbox validated the harness but showed that isolated elementwise tasks are poor standalone performance targets. The project then moved to an internal fused8 benchmark: `bias_relu`, `sigmoid`, `add_relu`, `residual`, `bias_gelu`, `row_sum`, `layernorm`, and `rmsnorm`. These fused workloads are better aligned with Triton launch amortization and realistic kernel-generation behavior.

## 5. Methods

Fused8 candidates expose `forward(*args)`. Official KernelBench tasks now materialize one seeded `get_init_inputs()` snapshot and construct persistent reference `Model` and candidate `ModelNew` instances from it before verification and timing. Every official `Model` task rejects free functions, including tasks with an empty `state_dict()`. Current benchmarking rotates eager, candidate, and compile order across three same-process sessions, reports session-level speedups, and materializes compilation before runtime sampling. The AST policy rejects high-level Torch compute, unsafe/import-time calls, task imports, and false Triton claims, but it is not an operating-system sandbox. Historical KernelBench records predate these corrections and are not interpreted as model or performance evidence.

## 6. Model and Template Baselines

- Deterministic templates sweep block size, warps, stages, allocation policy, contiguity policy, and shape specialization.
- Gemini rigorous fused8 used 24 generated candidates and verified 23/24.
- OpenAI mini rigorous fused8 used 24 generated candidates and verified 12/24.
- Configured model strings are audited in `reports/model_identifier_audit.md`: fused8 Gemini and KernelBench Gemini use `gemini-3.1-flash-lite`, and OpenAI mini uses `gpt-5.4-mini`. Provider-returned model-version fields are not preserved for every run.
- Earlier Gemini template-guided, OpenAI, GPT-5.5, and Qwen runs are legacy timing context, not primary paper-facing results.
- Qwen 7B local zero-shot was weak under the cheap fused8 protocol; Qwen 14B was not evaluated because serving failed due disk/cache capacity.

## 7. Results

Provenance note: the paper-facing rigorous fused8 numbers are from RunPod artifacts. `reports/artifact_index.md` records which canonical artifacts are present locally.

### Three-Task Conclusion

| Task | Best single-run speedup | Repeatability outcome | Final conclusion |
| --- | --- | --- | --- |
| vector_add | 0.692 | repeat median 0.483x | poor standalone target; launch overhead dominates |
| relu | 0.812 | repeat median 0.512x | poor standalone target; use only as harness check |
| bias_relu | 1.017 | repeat median 0.705x | single-run win was not stable; useful as fused-task seed |

### Fused8 Deterministic Template Results

These are the current paper-facing deterministic-template numbers from the rigorous CUDA-event run `runs/20260520_155839`. The older 2076-candidate template-wide table is legacy timing.
The fused8 suite uses one primary shape regime centered on `[4096, 1024]`, so the fused8 results characterize this controlled regime rather than scaling behavior.

| Task | Best single-run | Repeat median | Stable above eager | Above torch.compile |
| --- | --- | --- | --- | --- |
| bias_relu | 1.029 | 0.976 | no | yes |
| sigmoid | 0.998 | 0.934 | no | yes |
| add_relu | 0.947 | 0.938 | no | yes |
| residual | 1.140 | 1.023 | yes | yes |
| bias_gelu | 1.473 | 1.485 | yes | yes |
| row_sum | 0.790 | 0.674 | no | yes |
| layernorm | 0.843 | 0.791 | no | yes |
| rmsnorm | 1.674 | 1.452 | yes | yes |

### Model Comparison

| Baseline | Candidates | Verified | Median eager speedup | Median compile speedup | Uncertainty | Repeat-stable wins | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| template | 160 | 160/160 | 0.945 | 1.079 | not preserved | residual, bias_gelu, rmsnorm | strongest overall floor |
| Gemini | 24 | 23/24 | 0.923 | 1.863 | not preserved | bias_gelu, rmsnorm | strongest model correctness; stable wins below deterministic template medians |
| OpenAI mini | 24 | 12/24 | 0.888 | 1.835 | not preserved | residual | weaker correctness; one stable win over deterministic template repeat median |
| legacy model rows | various | various | legacy | legacy | legacy | legacy timing | historical context only; not primary paper-facing comparison |

## 8. Repeatability Analysis

Repeatability changed several conclusions. In the three-task sandbox, single-run wins for `bias_relu` did not survive repeat benchmarking. In rigorous fused8, deterministic `bias_relu` was also a single-run-only win: it reached 1.029x in the original run and fell to 0.976x repeat median. The top-1 LLM above-eager wins in the rigorous Gemini and OpenAI mini runs did survive repeatability, so the strongest broad LLM-fragility headline is not supported by this sample. Single-run wins remain useful search signals, but they are not sufficient evidence for benchmark claims.

The local artifact recovery pass imports the available KernelBench pilot and repair artifacts under `artifacts/runpod_imports/` and records SHA256 checksums. The three rigorous fused8 run directories are still missing locally, so full fused8 p25/p75, bootstrap intervals, per-session medians, and all-160-candidate single-run/repeat flip pairs remain unrecovered. The paper therefore reports the imported task-best `bias_relu` flip and does not claim a global fused8 flip rate.

### Stable Winners By Task

| Task | Stable winner | Source type | Repeat median | Uncertainty | Closest comparison | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| bias_relu | none | n/a | 0.976 | not preserved | template single-run 1.029 | single-run template win fell below eager |
| sigmoid | none | n/a | 0.997 | std 0.029, CV 0.030 | Gemini 0.997 | nearest model remained below eager |
| add_relu | none | n/a | 0.968 | std 0.003, CV 0.003 | Gemini 0.968 | nearest model remained below eager |
| residual | OpenAI mini | llm | 1.074 | std 0.048, CV 0.045 | template 1.023 | OpenAI mini is the only model-over-template stable win |
| bias_gelu | template | template | 1.485 | not preserved | Gemini 1.387 | template remains stronger than Gemini |
| row_sum | none | n/a | 0.674 | not preserved | Gemini 0.646 | all verified candidates below eager |
| layernorm | none | n/a | 0.791 | not preserved | Gemini 0.785 | all verified candidates below eager |
| rmsnorm | template | template | 1.452 | not preserved | Gemini 1.415 | template remains strongest by repeat median |

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

The most important result is not that one model wins. It is that correctness, single-run speed, repeat-stable speed, and compiler-baseline performance are separable. Gemini is strong on fused8 correctness, OpenAI mini finds the stable `residual` winner despite weaker correctness, and templates remain the strongest overall floor. Generated kernels may beat weak compiler-generated paths while still losing to library-specialized eager kernels. The evaluation layer is therefore the main contribution: generated kernels should be discussed with repeatability labels and compiler baselines, not isolated timing samples.

Historical KernelBench profiler and clock-recorded files are retained as debugging artifacts. They inherit the invalid reference lifecycle, so they do not support mechanism attribution or speedup persistence claims.

### Corrected workshop campaign

The corrected external study pinned official KernelBench commit
`423217d9fda91e0c2d67e4a43bf62f96f6d104f1`, selected 48 tasks without reading
candidate or timing fields, froze three Gemini candidates per task, and
separated screening from seven-process confirmation. Ten tasks produced a
fully valid frozen winner, all in the matrix-multiplication family. No winner
crossed the 2% eager margin. The median screening-to-confirmation ratio was
1.014 with task-bootstrap interval [0.915, 1.170]. Evaluator calibration and
lifecycle controls passed before screening; checksummed raw artifacts are under
`artifacts/workshop2026/`.

The separate all-candidate fused8 study confirmed every valid deterministic
candidate and found apparent and confirmed win rates of 1.0 for all tested
budgets. This negative multiplicity result is a boundary condition, not evidence
that search multiplicity is harmless in other settings.

A separate calibrated near-threshold stress test on an RTX A4500 froze 32
primary candidates after disjoint calibration. At `K=8`, the apparent win rate
was 0.75 and the independently confirmed rate was 0.50. The `bias_gelu` winner
fell from 1.0271x in screening to 1.0001x in confirmation. Median log optimism
was 0.007248 with four-task bootstrap interval `[-0.012395, 0.026614]`. This is
direct evidence of one screen-only promotion in the calibrated regime, but the
small task count and interval do not support a population claim.

## 12. Limitations

- The corrected KernelBench study is a deterministic feasible subset on one
  Tesla T4, not a random or full L1 sample; only matrix-multiplication tasks
  produced valid timed candidates.
- The fused8 benchmark uses one primary `[4096, 1024]` shape regime and should not be read as shape-scaling evidence.
- Small task sets and one GPU class per campaign; the KernelBench holdout used a
  Tesla T4 and the separate near-threshold stress test used an RTX A4500.
- Small rigorous model budgets: 24 generated candidates for Gemini and 24 for OpenAI mini.
- No Nsight or hardware-counter profiling yet.
- Historical KernelBench profiler and clock files are debugging artifacts only and are not used as paper evidence.
- Historical compile-time fields used incomplete accounting or were null; compile-cost amortization is not analyzed.
- No fine-tuning and no RL.
- API model behavior can change over time.
- No SOTA claim.

## 13. Historical KernelBench Adapter Audit

The preserved historical artifacts use the official KernelBench checkout at commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`. The affected evaluator recorded 3/20 one-shot verifications and 1/8 repair verifications, but these are audit observations only. The old prompt forced all candidates into free functions, violating the official `ModelNew` lifecycle for every task; parameterized tasks additionally could not own initialized state. Reference modules were reconstructed inside timed calls, and 9/20 one-shot sources fail the current strict policy. See `reports/kernelbench_adapter_audit.md` and `reports/tables/kernelbench_historical_policy_reaudit.csv`. The corrected workshop campaign supersedes these rows and reports no above-margin generated-kernel win.

## 14. Reproducibility Appendix

See `reports/reproducibility.md` for exact command flows. The table sources are in `reports/tables/`. Run artifacts that are not present in the local checkout are listed explicitly in `reports/artifact_index.md`.
