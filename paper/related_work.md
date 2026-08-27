# Related Work

## Positioning

OpenKernelForge sits between generated-kernel benchmarks, compiler baselines,
and systems-measurement methodology. KernelBench and CUDA-L1 focus on generated
kernel capability and improvement; Triton, Halide, TVM, and TorchInductor
provide compiler and DSL context; systems-measurement work motivates the
statistical caution. OpenKernelForge combines these threads as a
repeatability-aware evaluation and artifact-preservation layer. A compact
positioning table is kept in the appendix rather than the main body.

## LLM-Generated GPU Kernels And Kernel Agents

LLM-based kernel-generation systems attempt to translate tensor programs into CUDA or Triton implementations that compile, pass correctness tests, and improve runtime. KernelBench frames this as an evaluation problem: generated kernels must be judged by correctness and measured performance, not source plausibility [@kernelbench2025]. CUDA-L1 studies feedback-driven CUDA optimization as a model-improvement direction [@cudallm2025]. OpenKernelForge is different in scope. It is not a new model, training method, or leaderboard entry. It provides evaluation infrastructure for deciding which generated-kernel speedups are supported by preserved artifacts and repeatability-aware measurement.

## KernelBench And Generated-Kernel Benchmarks

KernelBench asks whether LLMs can write efficient GPU kernels across a broader suite of tasks [@kernelbench2025]. KernelBench-Verified subsequently shows that baseline configuration and narrow correctness distributions can overstate generated-kernel performance [@zhang2026kernelbenchverified]. Its hidden-test and TF32 analyses complement the present focus on task-state contracts, lifecycle isolation, repeatability, and artifact provenance. The present work does not report a validated KernelBench score. It uses preserved artifacts from a capped historical pilot to audit policy checks and evaluator behavior. The audit motivates the corrected `ModelNew` adapter and illustrates why evaluator validity must be established before model capability is inferred.

## Triton And GPU Program Synthesis

Triton provides a Python-embedded language and compiler for tiled neural-network kernels [@tillet2019triton]. It is a practical target for generated code because it exposes GPU programming concepts while remaining close to Python. OpenKernelForge also draws on a longer line of tensor and image-computation DSLs. Halide separates algorithm from schedule for image pipelines [@ragankelley2013halide], and TVM provides an optimizing compiler stack for deep-learning operators [@chen2018tvm]. OpenKernelForge does not propose a new compiler; it evaluates generated Triton programs against references, templates, and compiler baselines.

## Compiler Baselines

Generated kernels should be compared with compiler-produced code, not only eager PyTorch. PyTorch 2 introduced dynamic bytecode transformation and graph compilation exposed through `torch.compile` [@ansel2024pytorch2]. OpenKernelForge reports both speedup vs eager and speedup vs `torch.compile max-autotune` where available. Compile time is separated from runtime so that steady-state kernel claims are not inflated or penalized by compilation overhead.

## Measurement Reliability In GPU Benchmarking

Systems work has long shown that small timing and layout choices can change conclusions. Mytkowicz et al. show that measurement bias can produce wrong data without an obvious methodological error [@mytkowicz2009wrongdata]. Stabilizer argues for statistically sound performance evaluation under architectural nondeterminism [@curtsinger2013stabilizer]. Touati discusses statistical methodology for program-speedup claims [@touati2009statistical]. GPU timing adds warmup, launch overhead, cache state, memory allocation, synchronization, and compilation. CUDA events provide GPU-side elapsed-time measurement and are the timing primitive used by the rigorous OpenKernelForge path [@nvidiaCudaEvents]. The novelty is not CUDA timing itself. It is integrating repeatability labels, artifact preservation, static policy checks, and generated-kernel verification into one evaluation layer.

## Repair And Feedback Loops

Generated code often improves through feedback: verifier errors, compiler diagnostics, and runtime traces can become repair context. Self-refinement work studies iterative feedback loops for improving generated outputs [@madaan2023selfrefine]. The preserved OpenKernelForge repair pass is narrower: it provides one verifier error and the previous candidate once. Because that run used the affected KernelBench adapter, it is retained as workflow provenance rather than evidence of repair effectiveness.
