# Compile-Time Availability Notes

Historical KernelBench compile fields used obsolete accounting or are null. They are retained for schema auditing and are not used in the paper.

| Stage | Verified | Compile-time fields | Runtime-only fields | Notes |
| --- | ---: | ---: | ---: | --- |
| one-shot Gemini | 3 | 0 | 3 | historical fields use obsolete accounting or are null; not interpreted |
| repair1 | 1 | 0 | 1 | historical fields use obsolete accounting or are null; not interpreted |

The paper therefore does not analyze compile-cost amortization or deployment cost. This matters especially for `torch.compile max-autotune`, where compilation can be expensive relative to repeated runtime calls.
