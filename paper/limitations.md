# Limitations And Conclusion

## Limitations

The fused8 benchmark is internal and intentionally small. It is useful for controlled methodology development, but it is not a substitute for broad external evaluation. The fused8 suite uses one primary shape regime, centered on `[4096, 1024]` tensors. Its results characterize this controlled regime and should not be interpreted as evidence about scaling across batch sizes, feature dimensions, or sequence lengths.

The historical KernelBench pilot used 20 selected tasks, one Gemini candidate per task, and one repair pass over 8 failures. Its memory filter was a lower bound, and its deterministic feasible-subset selector was not a random L1 sample. More importantly, the free-function candidate contract violated the official `ModelNew` lifecycle, could not represent parameterized task state, and paired candidates with reference modules reconstructed inside timed calls. The paper therefore makes no KernelBench correctness or speedup claim from those runs. Corrected-adapter CUDA revalidation is outstanding.

The model budgets are small. Fused8 uses 24 candidates for Gemini and 24 for OpenAI mini. The historical KernelBench budget is documented only as generation provenance and is not interpreted as a model comparison.

OpenAI mini verifies 12/24 fused8 candidates, much lower than Gemini's 23/24. This is a real protocol result, but it should not be overread as a definitive model-family ranking: the prompt was shared across models and was not separately tuned for OpenAI mini, and API behavior, decoding defaults, and endpoint constraints can affect generated code.

Candidate search creates multiplicity. The deterministic template sweep evaluates many variants, and model runs select from generated candidates. Repeatability labels reduce the risk of promoting noisy single-run artifacts, but they do not eliminate selection bias or multiple-comparisons effects. The fused8 effects are reported only for their controlled setting; no family or model superiority claim is made.

The hardware campaign uses one RTX 5090 RunPod environment. API model versions and serving behavior may change. Historical KernelBench profiler and clock-recorded diagnostics inherit the affected reference lifecycle and are retained only as debugging artifacts. GPU and memory clocks were not locked for the original fused8 campaign; dynamic boost, power state, and thermal behavior may contribute to run-to-run variation. CUDA graphs were not used. For small-shape workloads, graph capture can reduce launch overhead and may change relative performance. Three same-process sessions provide a practical stability check rather than a high-powered variance estimate.

Repeatability thresholds are fixed implementation defaults rather than tuned constants. The legacy labeler uses `CV <= 0.10`; the rigorous session labeler uses `tau = 0.98`. The local artifacts do not preserve the per-session vectors needed for a full threshold-sensitivity analysis, so threshold robustness is not claimed.

Current code records compile wrapper construction plus the first synchronized materializing call separately from steady-state runtime. Historical compile-time fields used incomplete accounting or were null and are not interpreted. Compile-cost amortization remains outside this paper.

The historical repair artifacts remain useful for tracing prompts and verifier feedback, but the affected adapter prevents an effectiveness claim.

No model training, LoRA, RL, or execution-feedback optimization is performed. Qwen 7B is reported only as a zero-shot local baseline from earlier context, and Qwen 14B has no model-quality result because serving failed due infrastructure disk/cache capacity.

These constraints limit the paper to methodology and bounded characterization rather than leaderboard comparison.

## Conclusion

OpenKernelForge demonstrates an evaluation layer for generated Triton kernels. The central lesson is that correctness, single-run speed, repeat-stable speed, and compiler-baseline performance are different claims and should be reported separately.

In the internal fused8 study, deterministic templates remain the strongest overall floor, Gemini is highly correct, and OpenAI mini finds one repeat-stable win over the template on `residual`. The KernelBench audit supplies a separate lesson: external task-state contracts and baseline lifecycles must be validated before benchmark rows are treated as evidence.

The right claim is therefore methodological. The results suggest that repeatability-aware CUDA-event benchmarking is a useful default for generated-kernel evaluation: it changes which generated Triton speedups should be considered stable, preserves evidence for audit, and separates correctness, compiler speedup, eager speedup, and repeat-stable speedup. Broader campaigns are required before making benchmark-wide conclusions.
