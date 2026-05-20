# Related Work

## LLM-Generated GPU Kernels

Kernel-generation agents use language models to propose CUDA or Triton implementations for PyTorch workloads. The promising part is that models can often map high-level tensor programs into executable low-level kernels. The hard part is proving that the generated code is both correct and actually faster under fair measurement.

## KernelBench-Style Evaluation

KernelBench evaluates generated kernels for PyTorch workloads using correctness and performance metrics. It is the natural benchmark target for this project, but OpenKernelForge has not yet run a KernelBench result. The current internal fused8 benchmark is a smaller controlled environment for developing measurement and artifact discipline before moving to KernelBench L1.

## Triton Template Baselines

Triton makes concise GPU kernels easier to write, especially for elementwise fusion and small reductions. Deterministic templates are an important baseline because they show whether a model-generated candidate is better than a straightforward hand-structured implementation. In the current fused8 study, templates remain the strongest overall performance floor.

## Repeatability And Benchmark Variance

GPU runtime measurements can vary due to thermal state, clocking, cache effects, compilation, memory layout, and scheduler noise. A single successful timing run is not enough evidence for a speedup claim. OpenKernelForge treats repeatability as part of the benchmark result: top candidates should be remeasured across independent sessions before being called stable wins.

## Dataset Curation For Future Training

Generated-kernel logs can become training data, but only after review. Correct-but-slow candidates, failed-to-fixed repairs, template-vs-model optimization pairs, and repeat-stable fast candidates have different uses. OpenKernelForge preserves these distinctions so future SFT or optimization training does not mix unstable single-run wins with trusted targets.
