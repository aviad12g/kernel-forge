# Headline Clock Validation

These preserved rows came from the affected historical KernelBench adapter. They are retained as debugging artifacts only; the invalid reference lifecycle prevents treating the repeated timings or labels as current evidence. The fused8 candidates were unavailable and were not rechecked.

- Clock mode: clock-recorded
- CUDA available: True
- GPU: NVIDIA GeForce RTX 4090
- Torch: 2.9.1+cu128
- Triton: 3.5.1
- Corrected validation rows: 0
- Historical debug replays: 3
- Label changes among corrected validation rows: 0

| Task | Source | Old label | Validation label | Validation speedup | Status | Notes |
| --- | --- | --- | --- | ---: | --- | --- |
| bias_relu | template | SINGLE_RUN_ONLY_WIN | INSUFFICIENT_DATA | not available | unavailable | exact candidate artifact not preserved; Exact deterministic template candidate from runs/20260520_155839 is not present locally. |
| residual | OpenAI mini | REPEAT_STABLE_WIN | INSUFFICIENT_DATA | not available | unavailable | exact candidate artifact not preserved; Exact OpenAI mini candidate from runs/20260520_163607 is not present locally. |
| bias_gelu | template | REPEAT_STABLE_WIN | INSUFFICIENT_DATA | not available | unavailable | exact candidate artifact not preserved; Exact deterministic template candidate from runs/20260520_155839 is not present locally. |
| rmsnorm | template | REPEAT_STABLE_WIN | INSUFFICIENT_DATA | not available | unavailable | exact candidate artifact not preserved; Exact deterministic template candidate from runs/20260520_155839 is not present locally. |
| CrossEntropyLoss | Gemini one-shot | REPEAT_STABLE_WIN | REPEAT_STABLE_WIN | 1.856x | historical debug replay | repeated affected-adapter comparison; invalid as performance evidence |
| TripletMarginLoss | Gemini one-shot | REPEAT_STABLE_WIN | REPEAT_STABLE_WIN | 4.046x | historical debug replay | repeated affected-adapter comparison; invalid as performance evidence |
| KLDivLoss | Gemini repair1 | REPEAT_STABLE_WIN | REPEAT_STABLE_WIN | 1.834x | historical debug replay | repeated affected-adapter comparison; invalid as performance evidence |
