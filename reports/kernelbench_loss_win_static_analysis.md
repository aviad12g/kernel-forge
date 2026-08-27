# Historical KernelBench Loss-Candidate Source Audit

This report inspects preserved source from the affected historical adapter. The old task-state and reference lifecycle invalidate performance and mechanism attribution; speed fields are retained only to identify their source records.

| Task | Speedup vs eager | Speedup vs compile | Likely mechanism | Confidence | Caveat |
| --- | ---: | ---: | --- | --- | --- |
| CrossEntropyLoss | 1.992x | 2.895x | source contains row-wise log-sum-exp and target gather followed by a Torch mean | source present | historical source pattern only; invalid reference lifecycle prevents performance attribution |
| TripletMarginLoss | 4.176x | 3.208x | source contains one Triton kernel for distances and hinge loss followed by a Torch mean | source present | historical source pattern only; invalid reference lifecycle prevents performance attribution |
| KLDivLoss | 1.843x | 1.028x | source contains a Triton elementwise KL term plus Torch log and reduction | source present | historical source pattern only; invalid reference lifecycle prevents performance attribution; candidate still uses torch.log and torch sum outside Triton |

Interpretation: the source contains plausible fusion patterns, but the affected evaluator cannot establish that those patterns caused a valid speedup. Corrected candidate verification, timing, and profiling are required.
