# Workshop 2026 Near-Threshold Multiplicity Stress Test

Status: complete, checksum-verified, and derived without additional CUDA execution.

- Frozen primary candidates: 32.
- Recorded worker hours: 0.368.
- GPU: NVIDIA RTX A4500.
- Software: PyTorch 2.8.0+cu128; Triton 3.4.0; Python 3.12.3.
- Calibration was disjoint from primary timing and excluded from every estimate.
- Every selected candidate was inside the prespecified calibration window.

## Full-budget frozen winners

| Task | Candidate | Screen | Confirm | Label |
|---|---|---:|---:|---|
| bias_gelu | delay_10 | 1.0271x | 1.0001x | SCREEN_ONLY_WIN |
| bias_relu | delay_07 | 1.0369x | 1.0208x | CONFIRMED_WIN |
| residual_add_relu | delay_06 | 1.0188x | 1.0199x | BELOW_MARGIN |
| rmsnorm_small | delay_06 | 1.0214x | 1.0342x | CONFIRMED_WIN |

## Candidate-budget analysis

| K | Apparent win rate | Confirmed win rate | Median log optimism | 95% interval |
|---:|---:|---:|---:|---:|
| 1 | 0.1536 | 0.1243 | 0.003865 | [-0.012102, 0.025186] |
| 2 | 0.2917 | 0.2208 | 0.003865 | [-0.012395, 0.025662] |
| 3 | 0.4096 | 0.2947 | 0.004370 | [-0.012395, 0.026614] |
| 5 | 0.6014 | 0.4004 | 0.004370 | [-0.012395, 0.026614] |
| 8 | 0.7500 | 0.5000 | 0.007248 | [-0.012395, 0.026614] |
