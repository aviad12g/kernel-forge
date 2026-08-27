# Experiments

This section separates supported performance evidence from evaluator-audit artifacts. The internal fused8 study is the controlled comparison under CUDA-event timing. The historical KernelBench pilot is retained to document adapter and policy failures, not as an external performance result.

Notation: `compile` means `torch.compile max-autotune`; `template` means deterministic Triton template search; `residual`, `layernorm`, and `rmsnorm` abbreviate `residual_add_relu`, `layernorm_small`, and `rmsnorm_small`. Because the paper's claim is about measurement stability, fused8 tables report repeat medians with uncertainty summaries where available. Historical KernelBench interval fields are preserved only as audit metadata.

## Table 1: Measurement Protocol

| Study | Timing | Repeats | Sessions | Cache | Compiler | Source |
| --- | --- | ---: | ---: | --- | --- | --- |
| fused8 | CUDA events | 120 | 3 | 128 MB write | compile max-autotune | template, Gemini, OpenAI mini |
| Historical KernelBench L1 | CUDA events | 100/120 | 3 same-process loops | 128 MB write | compile max-autotune | affected one-shot + repair artifacts |

## Fused8 Results

The fused8 study is the controlled internal comparison: deterministic templates, Gemini, and OpenAI mini all use the same task suite and CUDA-event measurement protocol. The suite uses one primary shape regime centered on `[4096, 1024]`, so this section characterizes a controlled regime rather than scaling behavior. The main body keeps this table intentionally compact; per-task rows are in the appendix.

### Table 2: Fused8 Rigorous Summary

| Source | Candidates | Verified | Median vs eager | Median vs compile | Uncertainty | Stable wins |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| template | 160 | 160/160 | 0.945x | 1.079x | not preserved | `residual`, `bias_gelu`, `rmsnorm` |
| Gemini | 24 | 23/24 | 0.923x | 1.863x | not preserved | `bias_gelu`, `rmsnorm` |
| OpenAI mini | 24 | 12/24 | 0.888x | 1.835x | not preserved | `residual` |

The template sweep is the strongest overall floor. Gemini verifies 23/24 fused8 candidates and OpenAI mini verifies 12/24 under one shared prompt protocol. Candidate outputs are clustered within eight tasks, the prompt was not separately tuned for OpenAI mini, and candidate budgets differ from the template sweep. The descriptive Wilson intervals and raw counts therefore characterize this protocol; they are not independent-sample evidence for a general ranking of model families.

[[FIGURE:fused8_source_summary|Fused8 source summary. Templates provide the largest candidate pool and strongest floor; Gemini verifies nearly all fused8 candidates; OpenAI mini verifies fewer candidates but still contributes one stable win.|width=5.2]]

[[FIGURE:fused8_stable_speedups|Repeat-stable fused8 speedups by task/source. The 1.0x line marks eager parity; only repeat-stable above-eager wins are shown.|width=5.6]]

## Why Model Kernels Beat Compile But Often Lose To Eager

Gemini and OpenAI mini are strong relative to the compiler baseline on fused8, with median speedups of 1.863x and 1.835x versus `compile max-autotune`. Both remain below PyTorch eager at median. This is not contradictory: "beats compile" and "beats eager" are different claims. PyTorch eager often dispatches to highly optimized library kernels such as cuBLAS, cuDNN, or tuned ATen paths, while `torch.compile max-autotune` may generate compiler-produced code that is weaker for these small shapes. Generated Triton kernels may therefore be useful replacements for weak compiler-generated paths without being universal replacements for library-specialized eager kernels. This is a characterization finding, not a SOTA result.

## Case Study: A Deterministic Single-Run Win That Disappears

The cleanest motivation for repeatability is not an LLM failure. It is deterministic `bias_relu`. The template reached 1.029x in a single run but fell to a 0.976x repeat median, changing its label to `SINGLE_RUN_ONLY_WIN`. Because this candidate is deterministic, the flip is measurement/repeatability rather than model randomness. It shows that single-run timing can overstate performance even when candidate generation itself is fixed.

The imported task-best template table contains four task-best rows above eager in the single-run summary; one of those four falls below eager after repeat measurement. The full 160-candidate per-candidate flip frequency is not preserved in the local artifact package, so the paper treats `bias_relu` as an illustrative case rather than a global flip-rate estimate.

[[FIGURE:bias_relu_single_run_flip|The deterministic `bias_relu` template crosses eager in one run but falls below eager by repeat median. This is the clearest small example of why repeatability labels matter.|width=4.6]]

In contrast, the top-1 above-eager LLM wins in the rigorous fused8 runs survived repeatability. The broad claim that LLM-generated wins generally disappear is therefore not supported by this sample.

[[PAGEBREAK]]

## KernelBench Adapter Audit

The preserved KernelBench run used the official repository at commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`. It loaded 100 L1 tasks and selected 20 after a deterministic family-round-robin filter. The artifacts include candidates, policy results, verifier errors, timing records, profiler diagnostics, and one repair pass. They provide a detailed evaluator trace but do not support an external correctness or speed claim.

The post-hoc audit found two coupled defects. First, the historical prompt required a free `forward(*args)` receiving only `get_inputs()`, although every official `Model` task requires `ModelNew` initialized from `get_init_inputs()`. The mismatch affected all selected tasks and left parameterized convolution candidates with no valid route to the reference weights. Second, the historical reference wrapper constructed and transferred `Model` inside each reference call, while candidates were persistent. That lifecycle can contaminate eager and compiled timing.

The current `ast-v5` policy also rejects preserved one-shot sources that the old policy marked pass, including high-level Torch convolutions/reductions, in-place and alias/helper bypasses, and import-time model construction. The exact count is generated by the static policy re-audit, which does not execute candidates.

### Table 3: Historical Evaluator Output

| Stage | Attempted | Recorded verified | Recorded labels |
| --- | --- | --- | --- |
| Baseline | 20 selected tasks | 20 eager / 20 compile timed | provisional timing |
| Affected one-shot record | 20 candidates | 3/20 | CE, Triplet |
| Affected repair record | 8 selected repairs | 1/8 | KLDiv |
| Affected combined record | 20 tasks + 8 repairs | 4 recorded unique tasks | 3 recorded stable labels |

These numbers state what the affected evaluator recorded. They are not estimates of Gemini accuracy, family effects, repair effectiveness, or KernelBench performance. Historical profiler and clock-recorded diagnostics inherit the same reference lifecycle and are retained only as debugging artifacts.

[[FIGURE:kernelbench_pilot_funnel|Output funnel from the affected historical adapter. Counts describe preserved artifacts rather than corrected-adapter model accuracy.|width=6.4]]

[[FIGURE:kernelbench_failure_taxonomy|Failure categories recorded by the historical verifier. Contract-invalid parameterized candidates make this a diagnostic taxonomy rather than a model-capability estimate.|width=4.8]]

The corrected adapter materializes one seeded initialization-argument snapshot, constructs persistent reference `Model` and candidate `ModelNew` instances before warmup and timing, rejects free functions for all official `Model` tasks, rotates measurement order, materializes compilation before runtime sampling, and reports session-level summaries. Corrected CUDA revalidation is required before KernelBench rows return to the supported-results section.

## Interpretation

The supported evidence yields three bounded findings. First, deterministic templates are a strong floor in fused8. Second, compiler-baseline wins and eager wins are different claims. Third, repeatability-aware benchmarking prevents overclaiming by separating correctness, single-run speed, stable speed, and compiler-baseline comparison. The KernelBench audit adds a methodological warning: adapter contracts and baseline lifecycles must be validated before external rows are promoted.

Fused8 is the controlled performance study. KernelBench is currently an evaluator-audit case study. The corrected adapter is implemented and CPU-tested, but its GPU results are not yet part of the paper.

Stable-win counts are descriptive. Counts such as three template wins, two Gemini wins, or one OpenAI mini win identify candidates that survive the protocol under a fixed budget; they are too small to support statistically significant model rankings.

The paper's central contribution is not a new model or leaderboard result. It is an evaluation layer that makes generated-kernel claims more auditable.

### What This Paper Claims And Does Not Claim

| Supported claim | Not claimed |
| --- | --- |
| Repeatability-aware evaluation changes which wins are stable. | No state of the art generated-kernel performance claim. |
| Templates are a useful fused8 performance floor. | No benchmark-wide KernelBench result or estimate. |
| Historical artifacts expose adapter and policy failure modes. | No validated KernelBench accuracy, speedup, family-effect, or repair-effectiveness estimate from the affected runs. |
| Corrected task contracts are required before external rows are promoted. | No proof that LLM speedups are broadly repeatability-fragile. |
| Compiler-baseline and eager-baseline speedups are distinct claims. | No deployment-cost or compile-amortization analysis. |
