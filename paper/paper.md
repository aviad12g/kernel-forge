# OpenKernelForge: Repeatability-Aware Evaluation for LLM-Generated Triton Kernels

# Abstract

Generated GPU kernels require evidence of correctness and stable speed, not plausible source alone. OpenKernelForge is an artifact-preserving evaluation harness for LLM-generated Triton kernels: it combines static policy checks, correctness verification, CUDA-event timing, cache-state perturbation, repeated measurement sessions, session-level uncertainty summaries, and eager and compiler baselines. On a controlled internal fused8 suite, Gemini and OpenAI mini outperform `torch.compile max-autotune` at median, by 1.863x and 1.835x, while remaining below PyTorch eager at median; deterministic templates provide the strongest overall floor. A deterministic `bias_relu` candidate also changes from a 1.029x single-run win to a 0.976x repeat median. A post-hoc audit of the preserved KernelBench pilot found that its free-function adapter violated the official `ModelNew` lifecycle, could not represent parameterized task state, and reconstructed reference modules inside timed calls. We therefore retain those artifacts as an evaluator-audit case study but exclude their correctness rates and speedups from supported claims pending corrected CUDA revalidation. The contribution is methodological: generated-kernel evaluation should distinguish correctness, compiler speedup, eager speedup, and repeat-stable speedup while preserving enough evidence to audit the evaluator itself.

# Introduction

LLMs and coding agents increasingly emit low-level GPU code from high-level tensor programs. For this to be useful, generated kernels must be correct, inspectable, and faster under a measurement protocol that survives reruns. Correctness alone is insufficient: a kernel can pass tests and still be slower than PyTorch eager or `torch.compile`. A single timing run is also insufficient: GPU measurements are affected by warmup, lazy compilation, cache state, synchronization, clock behavior, and sampling variance.

The risk is amplified in generated-kernel systems because they search over candidates. If the selection signal is noisy, the system can optimize toward measurement artifacts rather than real runtime improvements. A candidate selected from many attempts may appear fast because it hit a favorable measurement condition. Without repeatability, that candidate can become a false win in a report or dataset.

OpenKernelForge addresses this evaluation problem. It is not primarily a kernel generator; it is an artifact-preserving evaluation layer. It treats each candidate kernel as a research artifact: the prompt, raw model output, extracted Python source, policy result, verifier result, benchmark samples, repeatability label, and dataset row are preserved together. The harness compares generated candidates against PyTorch eager, `torch.compile max-autotune`, and deterministic Triton templates that serve as an explicit performance floor.

This paper reports a controlled internal fused8 study and an audit of a capped KernelBench L1 pilot. The fused8 study compares deterministic templates, Gemini, and OpenAI mini under one CUDA-event protocol. The KernelBench artifacts cover one Gemini candidate per task and one repair pass, but the implementation audit found task-contract and reference-lifecycle defects. We retain that pilot as an evaluator-audit case study rather than supported external performance evidence.

The fused8 study uses one primary shape regime centered on `[4096, 1024]`, so it characterizes a controlled operating point rather than scaling across batch sizes, feature dimensions, or sequence lengths. Model identifiers are reported as configured API strings; the local artifact package preserves configured strings, but not every provider-returned model-version field.

The two components answer different questions. The fused8 study asks how candidate sources compare when tasks and timing are controlled. The KernelBench audit asks whether the external adapter itself satisfies official state and timing contracts. This distinction matters because an evaluation layer must expose defects in its own baselines and task adaptation.

# Contributions

- A repeatability-aware measurement protocol for generated Triton kernels using CUDA events, warmup, cache-state perturbation, independent sessions, uncertainty summaries, compile/runtime separation, and `torch.compile max-autotune`.
- An artifact-preserving verifier pipeline that records prompts, responses, extracted source, static policy checks, correctness traces, timing summaries, repeatability labels, and dataset metadata.
- A controlled fused8 study showing why deterministic templates and repeatability labels are necessary baselines for generated-kernel claims.
- A corrected KernelBench adapter contract that initializes persistent reference `Model` and candidate `ModelNew` modules outside timed regions and rejects state-incompatible candidates.
- An artifact-backed audit explaining why the historical KernelBench pilot requires corrected revalidation instead of silent relabeling.

# Claim Boundaries

The supported thesis is narrow: repeatability-aware CUDA-event benchmarking changes which generated Triton kernel speedups should be treated as stable and should be part of the evaluation layer before making performance claims.

The paper does not claim that LLM-generated speedups generally disappear under repeatability. In the rigorous fused8 runs, the top-1 above-eager LLM wins survived repeatability. Repeatability still matters because it downgraded deterministic `bias_relu` from a single-run win to below eager, and it gives a conservative label for generated candidates before they enter reports or datasets.

The historical KernelBench evaluator recorded 3/20 one-shot verifications and 1/8 repair verifications. These are audit observations, not evidence of Gemini accuracy or KernelBench speed: the old candidate contract excluded parameterized state, and the reference lifecycle contaminated timing. The corrected adapter is implemented and CPU-tested; corrected CUDA revalidation remains outstanding.

Imported fused8 summaries use legacy repeatability labels because those runs predate preservation of the current session-label inputs. Historical KernelBench files contain rigorous-label fields, but those labels remain provisional because the surrounding adapter was invalid. Current runs use the corrected rigorous labeler and session summaries.
