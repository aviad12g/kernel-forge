# Workshop 2026 Lifecycle-Ablation Uncertainty

This report uses only the 24 preserved process rows from the completed control. It does not rerun CUDA work.

Source: `artifacts/workshop2026/lifecycle_ablation/lifecycle_ablation.csv`

| Metric | Median | Process-row IQR | Task-cluster bootstrap 95% interval |
|---|---:|---:|---:|
| host lifecycle inflation | 1.053227 | [0.997971, 1.114237] | [0.950311, 1.556548] |
| enclosing event inflation | 1.000097 | [0.991857, 1.001912] | [0.992248, 1.212250] |

The IQR describes dispersion across process-level medians. The bootstrap resamples the eight tasks as clusters and retains the three process rows associated with each sampled task. The broad cluster intervals reflect task heterogeneity and are not evidence of a population-wide effect.
