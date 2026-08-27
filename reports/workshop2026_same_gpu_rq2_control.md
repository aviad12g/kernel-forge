# Same-GPU RQ2 Control

This control reruns the frozen easy deterministic candidate grid on the RTX A4500 used for the near-threshold study. It removes the earlier cross-GPU comparison without changing either primary result.

| K | Original T4 apparent | Original T4 confirmed | A4500 apparent | A4500 confirmed | A4500 median optimism (log) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | -0.000332 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | -0.000317 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | -0.000308 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | -0.000300 |
| 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | -0.000291 |
| 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | -0.000112 |

At K=20, the A4500 apparent and confirmed win rates are 1.0000 and 1.0000. This is a bounded hardware-control result for the four-task easy grid, not a broader claim about model-generated candidates.
