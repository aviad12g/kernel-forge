# KernelBench Interpretation Notes

This note summarizes artifacts from the affected historical KernelBench adapter. It does not execute candidates. Counts and timings are audit metadata, not model-accuracy or performance evidence.

## Family-Level Outcomes

| Family | Selected | One-shot verified | One-shot stable | Repair attempted | Repair verified | Combined correct | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| convolution | 7 | 0 | 0 | 0 | 0 | 0 | obsolete free-function contract also omitted reference state |
| matmul | 7 | 1 | 0 | 6 | 0 | 1 | affected evaluator recorded one verification and no stable label |
| pooling | 3 | 0 | 0 | 1 | 0 | 0 | affected evaluator recorded no verification |
| loss | 3 | 2 | 2 | 1 | 1 | 3 | affected evaluator recorded its stable labels in this family |

## Historical Loss-Candidate Source Patterns

Source patterns are listed for auditability. The invalid reference lifecycle prevents mechanism attribution from the historical profiler or timing rows.

| Task | Speedup vs eager | Likely mechanism | Caveat |
| --- | ---: | --- | --- |
| CrossEntropyLoss | 1.992x | source contains row-wise log-sum-exp and target gather followed by a Torch mean | historical source pattern only; invalid reference lifecycle prevents performance attribution |
| TripletMarginLoss | 4.176x | source contains one Triton kernel for distances and hinge loss followed by a Torch mean | historical source pattern only; invalid reference lifecycle prevents performance attribution |
| KLDivLoss | 1.843x | source contains a Triton elementwise KL term plus Torch log and reduction | historical source pattern only; invalid reference lifecycle prevents performance attribution; candidate still uses torch.log and torch sum outside Triton |

## Repairability

High repairability is a historical selection heuristic, not evidence that repair is effective.

## Eager and Compile Baselines

Eager-path notes are qualitative. Historical compile fields use obsolete accounting or are null and are not interpreted.

## Memory Filtering

The historical filter was a lower-bound selection heuristic, not a complete peak-memory estimate.
